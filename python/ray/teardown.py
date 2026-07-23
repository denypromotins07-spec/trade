"""
Ray Cluster Teardown

Graceful Ray cluster teardown logic that forcibly kills zombie worker processes 
and releases shared memory plasma locks to prevent boot failures.

Ensures clean shutdown even under error conditions.
"""

import os
import sys
import time
import signal
import logging
import subprocess
from typing import List, Optional, Set
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RayTeardownManager:
    """
    Manages graceful Ray cluster teardown.
    
    Responsibilities:
    1. Stop Ray head and worker nodes gracefully
    2. Kill zombie processes that don't respond to SIGTERM
    3. Release shared memory plasma locks
    4. Clean up temporary directories
    5. Prevent boot failures from stale state
    """
    
    def __init__(self, timeout_seconds: int = 30):
        """
        Initialize teardown manager.
        
        Args:
            timeout_seconds: Timeout for graceful shutdown before force kill
        """
        self.timeout_seconds = timeout_seconds
        self.cleaned_pids: Set[int] = set()
        self.released_locks: List[str] = []
    
    def get_ray_pids(self) -> List[int]:
        """
        Get all PIDs associated with Ray processes.
        
        Returns:
            List of process IDs
        """
        ray_pids = []
        
        try:
            # Try using psutil for accurate detection
            import psutil
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'ray' in cmdline.lower():
                        ray_pids.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
                    
        except ImportError:
            # Fallback to pgrep
            try:
                result = subprocess.run(
                    ['pgrep', '-f', 'ray'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    ray_pids = [int(pid) for pid in result.stdout.strip().split()]
            except Exception:
                pass
        
        return ray_pids
    
    def get_plasma_lock_files(self) -> List[Path]:
        """
        Find all plasma shared memory lock files.
        
        Returns:
            List of lock file paths
        """
        lock_patterns = [
            "/tmp/ray/*/session*/sockets/plasma*",
            "/dev/shm/ray*",
            "/tmp/rayplasma*",
        ]
        
        locks = []
        for pattern in lock_patterns:
            try:
                for path in Path('/').glob(pattern.lstrip('/')):
                    if path.exists():
                        locks.append(path)
            except PermissionError:
                continue
            except Exception as e:
                logger.debug(f"Error searching {pattern}: {e}")
        
        return locks
    
    def stop_ray_gracefully(self) -> bool:
        """
        Attempt graceful Ray shutdown using ray.stop().
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Attempting graceful Ray shutdown...")
        
        try:
            import ray
            
            if ray.is_initialized():
                ray.shutdown()
                logger.info("Ray shutdown completed")
                return True
            else:
                logger.info("Ray was not initialized")
                return True
                
        except ImportError:
            logger.warning("Ray module not available")
            return False
        except Exception as e:
            logger.error(f"Graceful shutdown failed: {e}")
            return False
    
    def send_signal_to_processes(
        self,
        pids: List[int],
        sig: int = signal.SIGTERM
    ) -> Set[int]:
        """
        Send signal to list of processes.
        
        Args:
            pids: List of process IDs
            sig: Signal to send (default SIGTERM)
            
        Returns:
            Set of successfully signaled PIDs
        """
        signaled = set()
        
        for pid in pids:
            if pid in self.cleaned_pids:
                continue
                
            try:
                os.kill(pid, sig)
                signaled.add(pid)
                logger.info(f"Sent signal {sig} to PID {pid}")
            except ProcessLookupError:
                # Process already dead
                self.cleaned_pids.add(pid)
            except PermissionError:
                logger.warning(f"Permission denied for PID {pid}")
            except Exception as e:
                logger.error(f"Failed to signal PID {pid}: {e}")
        
        return signaled
    
    def wait_for_processes(
        self,
        pids: List[int],
        timeout: float = 5.0
    ) -> Set[int]:
        """
        Wait for processes to terminate.
        
        Args:
            pids: List of process IDs to wait for
            timeout: Maximum wait time in seconds
            
        Returns:
            Set of PIDs that have terminated
        """
        terminated = set()
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            still_running = []
            
            for pid in pids:
                if pid in terminated or pid in self.cleaned_pids:
                    continue
                    
                try:
                    # Check if process exists
                    os.kill(pid, 0)
                    still_running.append(pid)
                except ProcessLookupError:
                    terminated.add(pid)
                    self.cleaned_pids.add(pid)
                except PermissionError:
                    # Process exists but we can't signal it
                    still_running.append(pid)
            
            if not still_running:
                break
                
            time.sleep(0.1)
        
        return terminated
    
    def force_kill_remaining(self, pids: List[int]) -> Set[int]:
        """
        Force kill remaining processes with SIGKILL.
        
        Args:
            pids: List of process IDs
            
        Returns:
            Set of successfully killed PIDs
        """
        killed = set()
        
        for pid in pids:
            if pid in self.cleaned_pids:
                continue
                
            try:
                os.kill(pid, signal.SIGKILL)
                killed.add(pid)
                self.cleaned_pids.add(pid)
                logger.warning(f"Force killed PID {pid}")
            except ProcessLookupError:
                self.cleaned_pids.add(pid)
            except Exception as e:
                logger.error(f"Failed to kill PID {pid}: {e}")
        
        return killed
    
    def release_plasma_locks(self) -> List[str]:
        """
        Release plasma shared memory locks.
        
        Returns:
            List of released lock paths
        """
        logger.info("Releasing plasma locks...")
        released = []
        
        lock_files = self.get_plasma_lock_files()
        
        for lock_path in lock_files:
            try:
                if lock_path.exists():
                    lock_path.unlink()
                    released.append(str(lock_path))
                    logger.info(f"Released lock: {lock_path}")
            except PermissionError:
                logger.warning(f"Permission denied for lock: {lock_path}")
            except Exception as e:
                logger.error(f"Failed to release lock {lock_path}: {e}")
        
        self.released_locks = released
        return released
    
    def cleanup_temp_directories(self) -> int:
        """
        Clean up Ray temporary directories.
        
        Returns:
            Number of directories cleaned
        """
        logger.info("Cleaning up temporary directories...")
        cleaned_count = 0
        
        temp_patterns = [
            "/tmp/ray",
            "/tmp/rayplasma*",
        ]
        
        for pattern in temp_patterns:
            try:
                temp_dir = Path(pattern)
                if temp_dir.exists():
                    # Remove directory tree
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    cleaned_count += 1
                    logger.info(f"Cleaned up: {temp_dir}")
            except Exception as e:
                logger.error(f"Failed to clean {pattern}: {e}")
        
        return cleaned_count
    
    def execute_teardown(self) -> bool:
        """
        Execute complete teardown sequence.
        
        Returns:
            True if teardown completed successfully
        """
        logger.info("=" * 60)
        logger.info("EXECUTING RAY CLUSTER TEARDOWN")
        logger.info("=" * 60)
        
        success = True
        
        # Step 1: Graceful shutdown
        self.stop_ray_gracefully()
        time.sleep(1)
        
        # Step 2: Get remaining Ray processes
        ray_pids = self.get_ray_pids()
        logger.info(f"Found {len(ray_pids)} Ray processes")
        
        if ray_pids:
            # Step 3: Send SIGTERM
            self.send_signal_to_processes(ray_pids, signal.SIGTERM)
            
            # Step 4: Wait for graceful termination
            terminated = self.wait_for_processes(ray_pids, timeout=self.timeout_seconds)
            remaining = [p for p in ray_pids if p not in terminated]
            
            # Step 5: Force kill remaining
            if remaining:
                logger.warning(f"{len(remaining)} processes require force kill")
                self.force_kill_remaining(remaining)
                time.sleep(1)
        
        # Step 6: Release plasma locks
        self.release_plasma_locks()
        
        # Step 7: Clean temp directories
        self.cleanup_temp_directories()
        
        # Verify cleanup
        remaining_pids = self.get_ray_pids()
        if remaining_pids:
            logger.error(f"{len(remaining_pids)} Ray processes still running!")
            success = False
        else:
            logger.info("All Ray processes terminated")
        
        logger.info("=" * 60)
        logger.info(f"TEARDOWN {'COMPLETED' if success else 'FAILED'}")
        logger.info(f"PIDs cleaned: {len(self.cleaned_pids)}")
        logger.info(f"Locks released: {len(self.released_locks)}")
        logger.info("=" * 60)
        
        return success


def main():
    """Main entry point for Ray teardown."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Graceful Ray cluster teardown')
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Timeout for graceful shutdown (seconds)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip graceful shutdown, force kill immediately'
    )
    
    args = parser.parse_args()
    
    # Create manager
    manager = RayTeardownManager(timeout_seconds=args.timeout)
    
    if args.force:
        logger.info("Force mode enabled, skipping graceful shutdown")
        # Get and kill all Ray processes immediately
        pids = manager.get_ray_pids()
        manager.force_kill_remaining(pids)
        manager.release_plasma_locks()
        manager.cleanup_temp_directories()
        success = len(manager.get_ray_pids()) == 0
    else:
        success = manager.execute_teardown()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
