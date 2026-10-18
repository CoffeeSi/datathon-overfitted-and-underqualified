"""Causal, conservative service-history hypothesis. No labels or legal facts at inference."""
import math

from .primitive_data import CLASSES


class CashSequence:
    """Receipt -> customer departure -> cash persists; missing history abstains.

    Outputs are research candidates: neither ownership continuity nor accounting
    completion can be proved by these three visual primitives.
    """

    def __init__(self, positive_threshold=.7, negative_threshold=.3, dwell_sec=2.,
                 max_gap_sec=2., receipt_ttl_sec=120.):
        values = (positive_threshold, negative_threshold, dwell_sec, max_gap_sec, receipt_ttl_sec)
        if (not all(math.isfinite(v) for v in values) or not 0 <= negative_threshold < positive_threshold <= 1
                or min(dwell_sec, max_gap_sec, receipt_ttl_sec) <= 0):
            raise ValueError("Invalid cash sequence thresholds/timing")
        self.high, self.low = positive_threshold, negative_threshold
        self.dwell, self.max_gap, self.ttl = dwell_sec, max_gap_sec, receipt_ttl_sec
        self.last_time = None
        self.receipt = None
        self.absent_since = None
        self.cash_since = None
        self.emitted = False
        self.events = []
        self.unresolved = []

    def _reset_episode(self):
        self.receipt = self.absent_since = self.cash_since = None
        self.emitted = False

    def update(self, time_sec, probabilities):
        if (not math.isfinite(time_sec) or time_sec < 0 or (self.last_time is not None and time_sec <= self.last_time)
                or set(probabilities) != set(CLASSES)
                or any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities.values())):
            raise ValueError("Invalid or unordered primitive observation")
        if self.last_time is not None and time_sec - self.last_time > self.max_gap:
            self._reset_episode()
        self.last_time = time_sec
        if self.receipt and time_sec - self.receipt["time_sec"] > self.ttl:
            self._reset_episode()
        state = {k: (1 if v >= self.high else 0 if v <= self.low else None) for k, v in probabilities.items()}
        client, cash, handover = (state[k] for k in CLASSES)
        # Start a new episode only with observed receipt while customer is at counter.
        # Repeated overlapping positive windows don't extend receipt TTL indefinitely.
        if client == 1 and handover == 1 and self.receipt is None:
            self.receipt = dict(time_sec=time_sec, confidence=probabilities[CLASSES[2]])
        if client == 0:
            if self.absent_since is None:
                self.absent_since = time_sec
        else:
            # Unknown is not absence; a returning/new customer ends this departure segment.
            self.absent_since = None
            self.cash_since = None
        if client == 0 and cash == 1:
            if self.cash_since is None:
                self.cash_since = time_sec
            if time_sec - self.cash_since >= self.dwell and not self.emitted:
                if self.receipt is not None:
                    event = dict(event_type="unsecured_accepted_cash", timestamp_sec=time_sec,
                                 receipt_sec=self.receipt["time_sec"], departure_sec=self.absent_since,
                                 cash_persistent_since_sec=self.cash_since,
                                 confidence=min(self.receipt["confidence"], probabilities[CLASSES[1]],
                                                1 - probabilities[CLASSES[0]]),
                                 status="research_candidate", requires_review=True,
                                 limitations="Cash identity/ownership, completed operation and legal applicability are unverified.")
                    self.events.append(event)
                else:
                    self.unresolved.append(dict(timestamp_sec=time_sec, reason="cash_visible_without_observed_receipt"))
                self.emitted = True
        elif client == 0 and cash == 0:
            self._reset_episode()  # client departed and surface clear; no outstanding visible cash
        elif cash is None:
            self.cash_since = None  # interrupted evidence does not satisfy dwell
        # End an emitted episode when the visible cash disappears, including while next client is present.
        if self.emitted and cash == 0:
            self._reset_episode()
        return dict(time_sec=time_sec, state=state, receipt_sec=self.receipt["time_sec"] if self.receipt else None)
