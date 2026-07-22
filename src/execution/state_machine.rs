//! src/execution/state_machine.rs
//!
//! Rigorous Finite State Machine for Order Lifecycle Management.
//!
//! This module implements a zero-allocation FSM for tracking complex order
//! lifecycles including partial fills, cancellations, rejections, and Binance
//! WebSocket execution report stream handling with sequence ID validation.
//!
//! Features:
//! - Zero-Allocation Transitions: Enum-based state representation.
//! - Sequence Validation: Ensures correct ordering of execution reports.
//! - Partial Fill Tracking: Accurate quantity and fee accounting.
//! - Binance Compatible: Matches Binance execution report semantics.

use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::atomic::{AtomicU64, Ordering};

/// Order states in the lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderState {
    /// Order submitted but not yet acknowledged by exchange.
    PendingNew,
    /// Order accepted and resting on book.
    New,
    /// Order partially filled.
    PartiallyFilled,
    /// Order completely filled.
    Filled,
    /// Cancel request sent, awaiting confirmation.
    PendingCancel,
    /// Order cancelled.
    Cancelled,
    /// Order rejected by exchange.
    Rejected,
    /// Order expired (e.g., IOC/FOK not filled).
    Expired,
}

impl OrderState {
    /// Check if state is terminal (no further transitions possible).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            OrderState::Filled
                | OrderState::Cancelled
                | OrderState::Rejected
                | OrderState::Expired
        )
    }

    /// Check if order is actively resting on book.
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            OrderState::New | OrderState::PartiallyFilled | OrderState::PendingNew
        )
    }
}

/// Execution report event from exchange.
#[derive(Debug, Clone)]
pub struct ExecutionReport {
    pub order_id: String,
    pub client_order_id: String,
    pub symbol: String,
    pub side: Side,
    pub order_type: OrderType,
    pub time_in_force: TimeInForce,
    pub price: f64,
    pub stop_price: Option<f64>,
    pub quantity: f64,
    pub filled_quantity: f64,
    pub remaining_quantity: f64,
    pub average_fill_price: f64,
    pub last_fill_price: f64,
    pub last_fill_quantity: f64,
    pub commission: f64,
    pub commission_asset: String,
    pub trade_id: Option<u64>,
    pub execution_type: ExecutionType,
    pub order_status: OrderState,
    pub reject_reason: Option<String>,
    pub timestamp_ns: u64,
    pub sequence_id: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Limit,
    Market,
    StopLoss,
    StopLossLimit,
    TakeProfit,
    TakeProfitLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForce {
    GTC, // Good Till Cancel
    IOC, // Immediate Or Cancel
    FOK, // Fill Or Kill
    GTX, // Post Only (Good Till Cross)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionType {
    New,
    Trade,      // Fill
    Canceled,
    Replaced,
    Rejected,
    Calculated, // Liquidation
    Expired,
}

/// Order state machine with full lifecycle tracking.
pub struct OrderStateMachine {
    pub order_id: String,
    pub client_order_id: String,
    pub symbol: String,
    pub side: Side,
    pub order_type: OrderType,
    pub time_in_force: TimeInForce,
    pub original_quantity: f64,
    pub price: f64,
    pub stop_price: Option<f64>,
    
    /// Current state
    pub state: OrderState,
    
    /// Fill tracking
    pub filled_quantity: f64,
    pub remaining_quantity: f64,
    pub average_fill_price: f64,
    pub total_commission: f64,
    pub commission_asset: String,
    
    /// Sequence tracking for order validation
    pub last_sequence_id: u64,
    pub fill_count: u32,
    
    /// Timestamps
    pub created_at_ns: u64,
    pub updated_at_ns: u64,
    pub filled_at_ns: Option<u64>,
}

impl OrderStateMachine {
    /// Create a new order in PendingNew state.
    pub fn new(
        order_id: String,
        client_order_id: String,
        symbol: String,
        side: Side,
        order_type: OrderType,
        time_in_force: TimeInForce,
        quantity: f64,
        price: f64,
        stop_price: Option<f64>,
        commission_asset: String,
    ) -> Self {
        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        Self {
            order_id,
            client_order_id,
            symbol,
            side,
            order_type,
            time_in_force,
            original_quantity: quantity,
            price,
            stop_price,
            state: OrderState::PendingNew,
            filled_quantity: 0.0,
            remaining_quantity: quantity,
            average_fill_price: 0.0,
            total_commission: 0.0,
            commission_asset,
            last_sequence_id: 0,
            fill_count: 0,
            created_at_ns: now_ns,
            updated_at_ns: now_ns,
            filled_at_ns: None,
        }
    }

    /// Process an execution report and transition state.
    /// Returns Ok(()) on success, Err on invalid transition or sequence.
    pub fn process_report(&mut self, report: &ExecutionReport) -> Result<(), StateTransitionError> {
        // Validate sequence ordering (prevent out-of-order processing)
        if report.sequence_id <= self.last_sequence_id && report.execution_type != ExecutionType::Trade {
            return Err(StateTransitionError::OutOfSequence);
        }
        
        // For trades, we allow same sequence (multi-fill scenarios)
        if report.execution_type == ExecutionType::Trade && report.sequence_id < self.last_sequence_id {
            return Err(StateTransitionError::OutOfSequence);
        }

        self.last_sequence_id = report.sequence_id;
        self.updated_at_ns = report.timestamp_ns;

        match report.execution_type {
            ExecutionType::New => self.handle_new(report)?,
            ExecutionType::Trade => self.handle_trade(report)?,
            ExecutionType::Canceled => self.handle_canceled(report)?,
            ExecutionType::Rejected => self.handle_rejected(report)?,
            ExecutionType::Expired => self.handle_expired(report)?,
            ExecutionType::Replaced => self.handle_replaced(report)?,
            ExecutionType::Calculated => self.handle_calculated(report)?,
        }

        Ok(())
    }

    fn handle_new(&mut self, report: &ExecutionReport) -> Result<(), StateTransitionError> {
        if !matches!(self.state, OrderState::PendingNew) {
            return Err(StateTransitionError::InvalidTransition {
                from: self.state,
                to: OrderState::New,
            });
        }
        self.state = OrderState::New;
        Ok(())
    }

    fn handle_trade(&mut self, report: &ExecutionReport) -> Result<(), StateTransitionError> {
        // Validate fill quantity
        if report.last_fill_quantity <= 0.0 {
            return Err(StateTransitionError::InvalidFillQuantity);
        }

        // Update fill statistics using weighted average
        let prev_notional = self.filled_quantity * self.average_fill_price;
        let new_notional = report.last_fill_quantity * report.last_fill_price;
        
        self.filled_quantity += report.last_fill_quantity;
        self.remaining_quantity = self.original_quantity - self.filled_quantity;
        
        if self.filled_quantity > 0.0 {
            self.average_fill_price = (prev_notional + new_notional) / self.filled_quantity;
        }

        // Update commission
        self.total_commission += report.commission;
        if !report.commission_asset.is_empty() {
            self.commission_asset = report.commission_asset.clone();
        }

        self.fill_count += 1;

        // Determine new state
        if self.remaining_quantity <= 0.0 || self.filled_quantity >= self.original_quantity * 0.9999 {
            self.state = OrderState::Filled;
            self.filled_at_ns = Some(report.timestamp_ns);
        } else {
            self.state = OrderState::PartiallyFilled;
        }

        Ok(())
    }

    fn handle_canceled(&mut self, report: &ExecutionReport) -> Result<(), StateTransitionError> {
        if !self.state.is_active() {
            return Err(StateTransitionError::InvalidTransition {
                from: self.state,
                to: OrderState::Cancelled,
            });
        }
        self.state = OrderState::Cancelled;
        self.remaining_quantity = report.remaining_quantity;
        Ok(())
    }

    fn handle_rejected(&mut self, report: &ExecutionReport) -> Result<(), StateTransitionError> {
        if self.state.is_terminal() {
            return Err(StateTransitionError::InvalidTransition {
                from: self.state,
                to: OrderState::Rejected,
            });
        }
        self.state = OrderState::Rejected;
        // Store rejection reason if provided
        // In production, this would be stored in a dedicated field
        Ok(())
    }

    fn handle_expired(&mut self, report: &ExecutionReport) -> Result<(), StateTransitionError> {
        if !self.state.is_active() {
            return Err(StateTransitionError::InvalidTransition {
                from: self.state,
                to: OrderState::Expired,
            });
        }
        self.state = OrderState::Expired;
        self.remaining_quantity = report.remaining_quantity;
        Ok(())
    }

    fn handle_replaced(&mut self, report: &ExecutionReport) -> Result<(), StateTransitionError> {
        // Handle order modification (price/quantity change)
        self.price = report.price;
        if let Some(sp) = report.stop_price {
            self.stop_price = Some(sp);
        }
        // Quantity might change for replace
        self.original_quantity = report.quantity;
        self.remaining_quantity = report.remaining_quantity;
        self.filled_quantity = report.filled_quantity;
        
        // State remains active
        if self.filled_quantity > 0.0 {
            self.state = OrderState::PartiallyFilled;
        } else {
            self.state = OrderState::New;
        }
        
        Ok(())
    }

    fn handle_calculated(&mut self, report: &ExecutionReport) -> Result<(), StateTransitionError> {
        // Handle liquidation/adl events
        self.state = OrderState::Filled;
        self.filled_at_ns = Some(report.timestamp_ns);
        Ok(())
    }

    /// Get current fill percentage.
    pub fn fill_percentage(&self) -> f64 {
        if self.original_quantity <= 0.0 {
            return 0.0;
        }
        (self.filled_quantity / self.original_quantity) * 100.0
    }

    /// Check if order can be cancelled.
    pub fn can_cancel(&self) -> bool {
        self.state.is_active() && !self.state.is_terminal()
    }

    /// Get unrealized PnL based on current mark price.
    pub fn unrealized_pnl(&self, mark_price: f64) -> f64 {
        if self.filled_quantity <= 0.0 {
            return 0.0;
        }
        
        match self.side {
            Side::Buy => (mark_price - self.average_fill_price) * self.filled_quantity,
            Side::Sell => (self.average_fill_price - mark_price) * self.filled_quantity,
        }
    }

    /// Get realized PnL (only valid when filled).
    pub fn realized_pnl(&self, exit_price: f64) -> f64 {
        if self.state != OrderState::Filled {
            return 0.0;
        }
        
        let gross_pnl = match self.side {
            Side::Buy => (exit_price - self.average_fill_price) * self.filled_quantity,
            Side::Sell => (self.average_fill_price - exit_price) * self.filled_quantity,
        };
        
        gross_pnl - self.total_commission
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StateTransitionError {
    InvalidTransition { from: OrderState, to: OrderState },
    OutOfSequence,
    InvalidFillQuantity,
    TerminalState,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_lifecycle() {
        let mut fsm = OrderStateMachine::new(
            "ORD_123".to_string(),
            "client_1".to_string(),
            "BTCUSDT".to_string(),
            Side::Buy,
            OrderType::Limit,
            TimeInForce::GTC,
            1.0,
            50000.0,
            None,
            "USDT".to_string(),
        );

        assert_eq!(fsm.state, OrderState::PendingNew);

        // Exchange accepts order
        let new_report = ExecutionReport {
            order_id: "ORD_123".to_string(),
            client_order_id: "client_1".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: Side::Buy,
            order_type: OrderType::Limit,
            time_in_force: TimeInForce::GTC,
            price: 50000.0,
            stop_price: None,
            quantity: 1.0,
            filled_quantity: 0.0,
            remaining_quantity: 1.0,
            average_fill_price: 0.0,
            last_fill_price: 0.0,
            last_fill_quantity: 0.0,
            commission: 0.0,
            commission_asset: "".to_string(),
            trade_id: None,
            execution_type: ExecutionType::New,
            order_status: OrderState::New,
            reject_reason: None,
            timestamp_ns: 1000000,
            sequence_id: 1,
        };

        fsm.process_report(&new_report).unwrap();
        assert_eq!(fsm.state, OrderState::New);

        // Partial fill
        let fill_report = ExecutionReport {
            order_id: "ORD_123".to_string(),
            client_order_id: "client_1".to_string(),
            symbol: "BTCUSDT".to_string(),
            side: Side::Buy,
            order_type: OrderType::Limit,
            time_in_force: TimeInForce::GTC,
            price: 50000.0,
            stop_price: None,
            quantity: 1.0,
            filled_quantity: 0.5,
            remaining_quantity: 0.5,
            average_fill_price: 50000.0,
            last_fill_price: 50000.0,
            last_fill_quantity: 0.5,
            commission: 0.5,
            commission_asset: "USDT".to_string(),
            trade_id: Some(1001),
            execution_type: ExecutionType::Trade,
            order_status: OrderState::PartiallyFilled,
            reject_reason: None,
            timestamp_ns: 2000000,
            sequence_id: 2,
        };

        fsm.process_report(&fill_report).unwrap();
        assert_eq!(fsm.state, OrderState::PartiallyFilled);
        assert_eq!(fsm.filled_quantity, 0.5);
        assert_eq!(fsm.fill_count, 1);
    }
}
