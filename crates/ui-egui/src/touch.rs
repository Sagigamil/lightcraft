//! Touch gestures egui doesn't recognise itself: a two-finger tap (both fingers down and up
//! again quickly, without moving). On touch screens it opens the photo's context menu in the
//! detail view, where a long press shows the original.

/// How far a finger may drift during a tap (points).
const SLOP: f32 = 12.0;
/// From the first finger down to the last finger up (seconds).
const MAX_TAP: f64 = 0.4;
/// More simultaneous touches than this is not a tap; also caps the per-gesture state.
const MAX_FINGERS: usize = 10;

#[derive(Clone, Debug, Default)]
pub struct TouchGestures {
    /// Fingers down now: (touch id, where it went down).
    down: Vec<(u64, egui::Pos2)>,
    /// Where each finger of the current gesture went down (also the lifted ones).
    starts: Vec<egui::Pos2>,
    started: f64,
    spoiled: bool,
    /// A two-finger tap that ended this frame: the centre between the fingers.
    pub two_finger_tap: Option<egui::Pos2>,
    /// More than one finger is down in the current gesture, or was in one that ended this frame:
    /// a pointer click now is part of it (a two-finger tap, a pinch), not a one-finger tap.
    pub multi: bool,
}

impl TouchGestures {
    /// Feed one frame's events (`time`: the frame's input time, seconds).
    pub fn frame(&mut self, time: f64, events: &[egui::Event]) {
        self.two_finger_tap = None;
        self.multi = self.starts.len() > 1;
        for e in events {
            if let egui::Event::Touch { id, phase, pos, .. } = e {
                self.touch(time, id.0, *phase, *pos);
            }
        }
    }

    fn touch(&mut self, time: f64, id: u64, phase: egui::TouchPhase, pos: egui::Pos2) {
        use egui::TouchPhase::*;
        match phase {
            Start => {
                // a missed End (the view was not drawn, the app paused) must not wedge the state
                if self.down.is_empty() || time - self.started > 2.0 {
                    *self = TouchGestures { started: time, ..Default::default() };
                }
                self.down.retain(|(i, _)| *i != id);
                if self.down.len() >= MAX_FINGERS {
                    self.spoiled = true;
                    return;
                }
                self.down.push((id, pos));
                self.starts.push(pos);
                self.multi |= self.starts.len() > 1;
                if self.starts.len() > 2 {
                    self.spoiled = true;
                }
            }
            Move => {
                if self.down.iter().any(|(i, s)| *i == id && s.distance(pos) > SLOP) {
                    self.spoiled = true;
                }
            }
            End | Cancel => {
                if phase == Cancel || self.down.iter().any(|(i, s)| *i == id && s.distance(pos) > SLOP) {
                    self.spoiled = true;
                }
                self.down.retain(|(i, _)| *i != id);
                if self.down.is_empty() {
                    if let [a, b] = self.starts[..]
                        && !self.spoiled
                        && time - self.started <= MAX_TAP
                    {
                        self.two_finger_tap = Some(egui::pos2((a.x + b.x) / 2.0, (a.y + b.y) / 2.0));
                    }
                    self.starts.clear();
                    self.spoiled = false;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::TouchPhase::*;
    use egui::pos2;

    fn t(id: u64, phase: egui::TouchPhase, x: f32, y: f32) -> egui::Event {
        egui::Event::Touch { device_id: egui::TouchDeviceId(0), id: egui::TouchId(id), phase, pos: pos2(x, y), force: None }
    }

    #[test]
    fn two_quick_still_fingers_are_a_tap_at_their_centre() {
        let mut g = TouchGestures::default();
        g.frame(0.0, &[t(1, Start, 100.0, 100.0)]);
        g.frame(0.02, &[t(2, Start, 200.0, 100.0)]);
        assert_eq!(g.two_finger_tap, None);
        g.frame(0.1, &[t(1, Move, 104.0, 101.0), t(2, End, 200.0, 100.0)]);
        assert_eq!(g.two_finger_tap, None, "one finger is still down");
        g.frame(0.15, &[t(1, End, 104.0, 101.0)]);
        assert_eq!(g.two_finger_tap, Some(pos2(150.0, 100.0)));
        g.frame(0.2, &[]);
        assert_eq!(g.two_finger_tap, None, "reported once");
    }

    #[test]
    fn pinches_slow_holds_one_or_three_fingers_are_not_taps() {
        let mut g = TouchGestures::default();
        // pinch
        g.frame(0.0, &[t(1, Start, 100.0, 100.0), t(2, Start, 200.0, 100.0)]);
        g.frame(0.05, &[t(1, Move, 80.0, 100.0), t(2, Move, 220.0, 100.0)]);
        g.frame(0.1, &[t(1, End, 80.0, 100.0), t(2, End, 220.0, 100.0)]);
        assert_eq!(g.two_finger_tap, None);
        // too slow
        g.frame(1.0, &[t(1, Start, 100.0, 100.0), t(2, Start, 200.0, 100.0)]);
        g.frame(1.6, &[t(1, End, 100.0, 100.0), t(2, End, 200.0, 100.0)]);
        assert_eq!(g.two_finger_tap, None);
        // one finger
        g.frame(2.0, &[t(1, Start, 100.0, 100.0)]);
        g.frame(2.1, &[t(1, End, 100.0, 100.0)]);
        assert_eq!(g.two_finger_tap, None);
        // three fingers
        g.frame(3.0, &[t(1, Start, 100.0, 100.0), t(2, Start, 200.0, 100.0), t(3, Start, 300.0, 100.0)]);
        g.frame(3.1, &[t(1, End, 100.0, 100.0), t(2, End, 200.0, 100.0), t(3, End, 300.0, 100.0)]);
        assert_eq!(g.two_finger_tap, None);
        // cancelled
        g.frame(4.0, &[t(1, Start, 100.0, 100.0), t(2, Start, 200.0, 100.0)]);
        g.frame(4.1, &[t(1, Cancel, 100.0, 100.0), t(2, End, 200.0, 100.0)]);
        assert_eq!(g.two_finger_tap, None);
        // and a real tap still works afterwards
        g.frame(5.0, &[t(1, Start, 100.0, 100.0), t(2, Start, 200.0, 100.0)]);
        g.frame(5.1, &[t(1, End, 100.0, 100.0), t(2, End, 200.0, 100.0)]);
        assert_eq!(g.two_finger_tap, Some(pos2(150.0, 100.0)));
    }

    #[test]
    fn missed_ends_and_endless_fingers_never_wedge_or_grow() {
        let mut g = TouchGestures::default();
        // a finger whose End never came, then a later tap
        g.frame(0.0, &[t(9, Start, 10.0, 10.0)]);
        g.frame(5.0, &[t(1, Start, 100.0, 100.0), t(2, Start, 200.0, 100.0)]);
        g.frame(5.1, &[t(1, End, 100.0, 100.0), t(2, End, 200.0, 100.0)]);
        assert_eq!(g.two_finger_tap, Some(pos2(150.0, 100.0)));
        // hostile: thousands of distinct touches
        let many: Vec<_> = (0..5000).map(|i| t(i, Start, 0.0, 0.0)).collect();
        g.frame(10.0, &many);
        assert!(g.down.len() <= MAX_FINGERS && g.starts.len() <= MAX_FINGERS);
    }
}
