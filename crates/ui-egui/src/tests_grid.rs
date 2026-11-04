//! Headless tests of the photo grid on a larger synthetic library (issue #35): unchanged frames
//! reuse the view's date runs, layout and stack index and visit only the rows near the screen,
//! while date headers, scrolling to the active photo, keyboard navigation, stacks and
//! crop/orientation changes keep working.

use std::time::Duration;

use lightcraft_catalog::{Op, Photo, PhotoId, Source, Stack, StackId};
use lightcraft_engine::Session;
use serde_json::json;

use crate::headless::Headless;
use crate::state::ViewMode;
use crate::{LightcraftApp, Services};

const T: Duration = Duration::from_secs(30);
const SETTLE: Duration = Duration::from_secs(120);
const N: u64 = 3000;

/// `N` photos (missing files: thumbnails fail fast), 40 per capture day, landscape except every
/// fifth; a three-photo stack every 100 photos.
fn library() -> Session {
    let mut s = Session::new();
    for i in 0..N {
        let (w, h) = if i % 5 == 4 { (4000, 6000) } else { (6000, 4000) };
        let path = format!("/lightcraft-grid-test/IMG_{i:05}.jpg");
        let mut p = Photo::new(PhotoId(i + 1), Source::File { path }, &format!("IMG_{i:05}.jpg"), "JPEG", w, h, "2026-01-01T00:00:00");
        let day = i / 40;
        p.captured = Some(format!("2026-{:02}-{:02}T{:02}:00:00", 12 - day / 28, 28 - day % 28, 8 + i % 40 / 4));
        s.catalog.apply(Op::AddPhoto { photo: Box::new(p) }).unwrap();
    }
    for k in 0..N / 100 {
        let b = k * 100 + 1;
        let stack = Stack { id: StackId(k + 1), photos: vec![PhotoId(b), PhotoId(b + 1), PhotoId(b + 2)], collapsed: false };
        s.catalog.apply(Op::AddStack { stack }).unwrap();
    }
    s
}

fn grid(view: ViewMode) -> Headless {
    let app = LightcraftApp::new(library(), Services { png: None, ..Default::default() });
    let mut h = Headless::new(app, [1200.0, 800.0], 1.0);
    h.app.ui.view = view;
    h.app.ui.thumb_size = 160.0;
    h.settle(SETTLE);
    h
}

fn widget(h: &Headless, id: &str) -> Option<egui::Rect> {
    h.app.widgets.iter().find(|(w, _)| w == id).map(|(_, r)| *r)
}

fn select(h: &mut Headless, id: PhotoId) {
    let r = h.request("engine.execute", json!({"command": "library.select", "params": {"ids": [id.0]}}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.settle(SETTLE);
}

/// The photo's cell is drawn inside the grid's canvas.
fn on_screen(h: &Headless, id: PhotoId) -> bool {
    let canvas = h.app.canvas_rect.expect("grid drawn");
    widget(h, &format!("thumb:{}", id.0)).is_some_and(|r| canvas.contains_rect(r))
}

/// Unchanged frames build nothing and look only at the cells near the screen.
fn assert_cheap_frames(h: &mut Headless) {
    let before = h.app.caches.grid_stats.clone();
    let frames = 10;
    for _ in 0..frames {
        h.step();
    }
    let s = &h.app.caches.grid_stats;
    assert_eq!(s.frames - before.frames, frames);
    assert_eq!(s.view_builds, before.view_builds, "date runs rebuilt on an unchanged frame");
    assert_eq!(s.layout_builds, before.layout_builds, "layout rebuilt on an unchanged frame");
    assert_eq!(s.stack_index_builds, before.stack_index_builds, "stack index rebuilt on an unchanged frame");
    let per_frame = (s.cells_visited - before.cells_visited) / frames;
    assert!(per_frame > 0 && per_frame < 300, "visited {per_frame} of {N} cells per frame");
}

#[test]
fn unchanged_grid_frames_reuse_the_view_and_visit_only_nearby_rows() {
    for view in [ViewMode::PhotoGrid, ViewMode::SquareGrid] {
        let mut h = grid(view);
        assert_eq!(h.app.caches.grid_stats.view_builds, 1, "{view:?}");
        assert_eq!(h.app.caches.grid_stats.stack_index_builds, 1, "{view:?}");
        assert_cheap_frames(&mut h);
        // far down the grid too
        let far = h.app.session.visible()[2500];
        select(&mut h, far);
        assert_cheap_frames(&mut h);
    }
}

#[test]
fn grid_headers_selection_scrolling_stacks_and_shape_changes() {
    let mut h = grid(ViewMode::PhotoGrid);
    let vis = h.app.session.visible().to_vec();
    // the newest day's header heads the grid
    assert!(widget(&h, "group:2026-12-28").is_some(), "{:?}", h.app.widgets.iter().filter(|(w, _)| w.starts_with("group:")).collect::<Vec<_>>());
    assert_eq!(h.app.grid_scroll, Some(0.0));
    // selecting a photo far down scrolls the grid to it; its day's header is drawn (pinned or not)
    let target = PhotoId(2001); // the top of a stack, captured on day 50
    let i = vis.iter().position(|x| *x == target).unwrap();
    select(&mut h, target);
    assert!(h.app.grid_scroll.unwrap() > 1000.0, "{:?}", h.app.grid_scroll);
    assert!(on_screen(&h, target), "the active photo is scrolled into view");
    // stacks: the badge on the stack's top and members
    assert!(widget(&h, &format!("stack:{}", target.0)).is_some(), "stack badge drawn");
    assert!(widget(&h, "stack:2002").is_some(), "stack member badge drawn");
    // keyboard: the next photo becomes active and stays in view
    h.request("ui.key", json!({"key": "right"}), T);
    h.settle(SETTLE);
    let next = vis[i + 1];
    assert_eq!(h.app.session.selection.active, Some(next));
    assert!(on_screen(&h, next));
    // the user's scroll position is kept on unchanged frames
    let y = h.app.grid_scroll;
    assert_cheap_frames(&mut h);
    assert_eq!(h.app.grid_scroll, y);
    // rotating a landscape photo re-lays the grid with its new (portrait) shape
    let landscape = vis[i + 2];
    let before = widget(&h, &format!("thumb:{}", landscape.0)).unwrap();
    assert!(before.width() > before.height(), "{before:?}");
    let layouts = h.app.caches.grid_stats.layout_builds;
    let r = h.request("engine.execute", json!({"command": "photo.rotateRight", "params": {"ids": [landscape.0]}}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.settle(SETTLE);
    assert!(h.app.caches.grid_stats.layout_builds > layouts, "a shape change re-lays the grid");
    let after = widget(&h, &format!("thumb:{}", landscape.0)).unwrap();
    assert!(after.height() > after.width(), "{after:?}");
    assert_cheap_frames(&mut h);
}

/// Selected photos whose thumbnails have not loaded (or can't) are still marked selected in the
/// justified grid (issue #298): the active one white, the others grey.
#[test]
fn unloaded_thumbnails_show_their_selection() {
    let mut h = grid(ViewMode::PhotoGrid);
    h.request("engine.execute", json!({"command": "library.selectAll"}), T);
    let img = h.snapshot(SETTLE);
    assert_eq!(h.view.ctx.pixels_per_point(), 1.0);
    let active = h.app.session.selection.active.expect("active");
    let canvas = h.app.canvas_rect.expect("grid drawn");
    let mut checked = 0;
    for (w, r) in h.app.widgets.iter().filter(|(w, _)| w.starts_with("thumb:")) {
        if !canvas.shrink(4.0).contains_rect(*r) || r.min.x < 6.0 {
            continue;
        }
        let id = w.trim_start_matches("thumb:").parse::<u64>().unwrap();
        let want = if id == active.0 { egui::Color32::WHITE } else { egui::Color32::from_gray(170) };
        let px = img[((r.min.x - 1.0) as usize, r.center().y as usize)];
        assert_eq!(px, want, "photo {id}");
        checked += 1;
    }
    assert!(checked >= 3, "checked {checked}");
}

fn press(pos: egui::Pos2) -> egui::Event {
    egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed: true, modifiers: egui::Modifiers::NONE }
}

fn release(pos: egui::Pos2) -> egui::Event {
    egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed: false, modifiers: egui::Modifiers::NONE }
}

fn touch_start(pos: egui::Pos2) -> egui::Event {
    egui::Event::Touch { device_id: egui::TouchDeviceId(0), id: egui::TouchId(0), phase: egui::TouchPhase::Start, pos, force: None }
}

fn touch(id: u64, phase: egui::TouchPhase, pos: egui::Pos2) -> egui::Event {
    egui::Event::Touch { device_id: egui::TouchDeviceId(0), id: egui::TouchId(id), phase, pos, force: None }
}

/// One frame of a finger going down, as egui-winit sends it on Android: the touch, then the
/// emulated pointer move and press, in the same frame.
fn finger_down(h: &mut Headless, p: egui::Pos2) {
    h.app.synthetic.extend([touch(0, egui::TouchPhase::Start, p), egui::Event::PointerMoved(p), press(p)]);
    h.step();
}

fn finger_move(h: &mut Headless, p: egui::Pos2) {
    h.app.synthetic.extend([touch(0, egui::TouchPhase::Move, p), egui::Event::PointerMoved(p)]);
    h.step();
}

fn finger_up(h: &mut Headless, p: egui::Pos2) {
    h.app.synthetic.extend([touch(0, egui::TouchPhase::End, p), release(p), egui::Event::PointerGone]);
    h.step();
}

/// A finger dragged upwards from `start` by `dy` points, in 20-point steps.
fn swipe_up(h: &mut Headless, start: egui::Pos2, dy: f32) {
    finger_down(h, start);
    let mut y = 0.0;
    while y < dy {
        y = (y + 20.0).min(dy);
        finger_move(h, egui::pos2(start.x, start.y - y));
    }
    finger_up(h, egui::pos2(start.x, start.y - dy));
}

/// A touch drag that starts on a photo scrolls the grid every time, also after the grid sat idle
/// between gestures (Android: scrolling "sometimes" did nothing). The press is hit-tested against
/// the previous frame's widgets, so the cells must not sense drags on a touch screen even on frames
/// without touch events, or they steal the drag from the scroll area.
#[test]
fn touch_drag_on_a_photo_scrolls_the_grid_after_idle_frames() {
    let mut h = grid(ViewMode::PhotoGrid);
    for gesture in 0..3 {
        let canvas = h.app.canvas_rect.expect("grid drawn");
        // a photo cell well inside the grid
        let start = h
            .app
            .widgets
            .iter()
            .filter(|(w, _)| w.starts_with("thumb:"))
            .map(|(_, r)| r.center())
            .find(|c| canvas.shrink(150.0).contains(*c))
            .expect("a photo in the middle of the grid");
        let s0 = h.app.grid_scroll.unwrap_or(0.0);
        swipe_up(&mut h, start, 120.0);
        let s1 = h.app.grid_scroll.unwrap_or(0.0);
        // desktop builds start in mouse mode: the very first touch only switches to touch mode
        // (Android and iOS start in touch mode)
        if gesture > 0 {
            assert!(s1 > s0 + 60.0, "gesture {gesture}: the grid scrolled ({s0} -> {s1})");
        }
        assert_eq!(h.app.ui.view, ViewMode::PhotoGrid, "gesture {gesture}: a scroll never opens the photo");
        // the grid sits idle (no input) between gestures; kinetic scrolling settles
        for _ in 0..120 {
            h.step();
        }
    }
}

/// After the grid sat idle, a single tap on a photo still opens it (and only after the release).
#[test]
fn touch_tap_after_idle_frames_opens_the_photo() {
    let mut h = grid(ViewMode::PhotoGrid);
    // a first gesture makes egui see the touch screen, then the grid sits idle
    let first_id = h.app.session.visible()[0].0;
    let first = widget(&h, &format!("thumb:{first_id}")).expect("first cell");
    swipe_up(&mut h, first.center() + egui::vec2(0.0, 200.0), 40.0);
    for _ in 0..120 {
        h.step();
    }
    let canvas = h.app.canvas_rect.expect("grid drawn");
    let (id, r) = h
        .app
        .widgets
        .iter()
        .filter(|(w, r)| w.starts_with("thumb:") && canvas.shrink(100.0).contains(r.center()))
        .map(|(w, r)| (w.trim_start_matches("thumb:").parse::<u64>().unwrap(), *r))
        .next()
        .expect("a visible photo");
    finger_down(&mut h, r.center());
    assert_eq!(h.app.ui.view, ViewMode::PhotoGrid, "the press alone does not open the photo");
    finger_up(&mut h, r.center());
    assert_eq!(h.app.ui.view, ViewMode::Detail, "a tap opens the photo");
    assert_eq!(h.app.session.selection.active, Some(PhotoId(id)));
}

/// Two fingers spreading apart enlarge the grid thumbnails; pinching them together shrinks them.
/// Neither opens a photo.
#[test]
fn two_finger_pinch_zooms_the_grid() {
    let mut h = grid(ViewMode::PhotoGrid);
    let c = h.app.canvas_rect.expect("grid drawn").center();
    let pinch = |h: &mut Headless, from: f32, to: f32| {
        let (a, b) = (c - egui::vec2(from, 0.0), c + egui::vec2(from, 0.0));
        h.app.synthetic.extend([touch(0, egui::TouchPhase::Start, a), egui::Event::PointerMoved(a), press(a)]);
        h.step();
        h.app.synthetic.extend([touch(1, egui::TouchPhase::Start, b)]);
        h.step();
        for k in 1..=8 {
            let d = from + (to - from) * k as f32 / 8.0;
            let (a, b) = (c - egui::vec2(d, 0.0), c + egui::vec2(d, 0.0));
            h.app.synthetic.extend([touch(0, egui::TouchPhase::Move, a), egui::Event::PointerMoved(a), touch(1, egui::TouchPhase::Move, b)]);
            h.step();
        }
        let (a, b) = (c - egui::vec2(to, 0.0), c + egui::vec2(to, 0.0));
        h.app.synthetic.extend([touch(1, egui::TouchPhase::End, b)]);
        h.step();
        h.app.synthetic.extend([touch(0, egui::TouchPhase::End, a), release(a), egui::Event::PointerGone]);
        h.step();
    };
    let t0 = h.app.ui.thumb_size;
    pinch(&mut h, 50.0, 150.0);
    let t1 = h.app.ui.thumb_size;
    assert!(t1 > t0 * 1.5, "spreading enlarges the thumbnails ({t0} -> {t1})");
    assert_eq!(h.app.ui.view, ViewMode::PhotoGrid, "a pinch never opens a photo");
    for _ in 0..30 {
        h.step();
    }
    pinch(&mut h, 150.0, 50.0);
    let t2 = h.app.ui.thumb_size;
    assert!(t2 < t1 / 1.5, "pinching shrinks the thumbnails ({t1} -> {t2})");
    assert_eq!(h.app.ui.view, ViewMode::PhotoGrid, "a pinch never opens a photo");
}

/// A single touch tap on a photo opens it in the detail view (Lightroom mobile), even though
/// egui-winit sends `PointerGone` right after the release (so `latest_pos` is already `None`).
#[test]
fn touch_tap_opens_detail_view() {
    let mut h = grid(ViewMode::PhotoGrid);
    let target = h.app.session.visible()[0];
    let r = widget(&h, &format!("thumb:{}", target.0)).expect("first cell is visible");
    let p = r.center();
    h.app.synthetic.push(press(p));
    h.step();
    h.app.synthetic.push(touch_start(p));
    h.step();
    h.app.synthetic.push(release(p));
    h.step();
    assert_eq!(h.app.ui.view, ViewMode::Detail, "a touch tap opens the detail view");
    assert_eq!(h.app.session.selection.active, Some(target));
}

/// Dragging on a photo (scrolling the grid) must not open the detail view.
#[test]
fn touch_drag_does_not_open_detail_view() {
    let mut h = grid(ViewMode::PhotoGrid);
    let target = h.app.session.visible()[0];
    let r = widget(&h, &format!("thumb:{}", target.0)).expect("first cell is visible");
    let start = r.center();
    let end = egui::pos2(start.x, start.y - 80.0);
    h.app.synthetic.push(press(start));
    h.step();
    h.app.synthetic.push(touch_start(start));
    h.step();
    h.app.synthetic.push(egui::Event::PointerMoved(end));
    h.step();
    h.app.synthetic.push(release(end));
    h.step();
    assert_eq!(h.app.ui.view, ViewMode::PhotoGrid, "a drag scrolls instead of opening the photo");
}

/// Tapping a control drawn over/beside the grid (the bottom bar, the tool strip) must act on
/// that control, not open the photo under it.
#[test]
fn touch_on_grid_chrome_does_not_open_the_photo() {
    for id in ["button:copySettings", "icon:edit"] {
        let mut h = grid(ViewMode::PhotoGrid);
        let r = widget(&h, id).unwrap_or_else(|| panic!("no {id}"));
        let p = r.center();
        h.app.synthetic.push(press(p));
        h.step();
        h.app.synthetic.push(touch_start(p));
        h.step();
        h.app.synthetic.push(release(p));
        h.step();
        assert_ne!(h.app.ui.view, ViewMode::Detail, "a tap on {id} opened the photo");
    }
}

/// A tap on an open menu item over the grid must run the menu command, not open the photo.
#[test]
fn touch_on_an_open_menu_item_does_not_open_the_photo() {
    let mut h = grid(ViewMode::PhotoGrid);
    let r = h.request("ui.clickWidget", json!({"id": "menu:Edit"}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.step();
    let item = widget(&h, "menu.item:library.selectAll").expect("Select All is in the open menu");
    let p = item.center();
    h.app.synthetic.push(press(p));
    h.step();
    h.app.synthetic.push(touch_start(p));
    h.step();
    h.app.synthetic.push(release(p));
    h.step();
    assert_ne!(h.app.ui.view, ViewMode::Detail, "a menu item tap opened the photo");
    assert!(h.app.session.selection.ids.len() > 1, "Select All still ran");
}

/// Deleting the active photo activates the next one (not the first) and leaves the grid
/// scrollable where the user was.
#[test]
fn deleting_a_photo_keeps_the_grid_in_place_and_scrollable() {
    let mut h = grid(ViewMode::PhotoGrid);
    let vis = h.app.session.visible_cloned();
    let target = vis[100];
    select(&mut h, target);
    h.settle(SETTLE);
    let before = h.app.grid_scroll.unwrap_or(0.0);
    let r = h.request("engine.execute", json!({"command": "photo.delete", "params": {"ids": [target.0]}}), T);
    assert_eq!(r["ok"], true, "{r}");
    assert_eq!(h.app.session.selection.active, Some(vis[101]), "the next photo is active");
    h.settle(SETTLE);
    let after = h.app.grid_scroll.unwrap_or(0.0);
    assert!((after - before).abs() < 100.0, "the grid did not jump to the top ({before} -> {after})");
    // and scrolling still works
    h.request("ui.move", json!({"x": 400.0, "y": 400.0}), T);
    let s0 = h.app.grid_scroll.unwrap_or(0.0);
    h.request("ui.scroll", json!({"dy": -400.0}), T);
    h.step();
    let s1 = h.app.grid_scroll.unwrap_or(0.0);
    assert_ne!(s1, s0, "the grid still scrolls after a delete ({s0} -> {s1})");
}

/// The user path: long-press/right-click a photo, choose Delete Photo. It deletes, activates the
/// next photo, closes the menu and leaves the grid usable (scrollable).
#[test]
fn deleting_from_the_context_menu_keeps_the_grid_usable() {
    let mut h = grid(ViewMode::PhotoGrid);
    let r = h.request("ui.resize", json!({"width": 1200.0, "height": 2000.0}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.settle(SETTLE);
    let vis = h.app.session.visible_cloned();
    let target = vis[100];
    select(&mut h, target);
    h.settle(SETTLE);
    let thumb = widget(&h, &format!("thumb:{}", target.0)).expect("thumb");
    let r = h.request("ui.click", json!({"x": thumb.center().x, "y": thumb.center().y, "button": "right"}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.step();
    let del = widget(&h, "context:delete").expect("Delete Photo in the context menu");
    let p = del.center();
    h.app.synthetic.push(press(p));
    h.step();
    h.app.synthetic.push(touch_start(p));
    h.step();
    h.app.synthetic.push(release(p));
    h.step();
    assert!(!h.app.session.visible().contains(&target), "the photo was deleted");
    assert_eq!(h.app.session.selection.active, Some(vis[101]), "the next photo is active");
    // the menu is closed and the grid still scrolls
    h.request("ui.move", json!({"x": 400.0, "y": 400.0}), T);
    let s0 = h.app.grid_scroll.unwrap_or(0.0);
    h.request("ui.scroll", json!({"dy": -400.0}), T);
    h.step();
    assert_ne!(h.app.grid_scroll.unwrap_or(0.0), s0, "the grid still scrolls after the menu delete");
}

/// The detail view of the first photo, with a touch screen already seen.
fn detail() -> (Headless, egui::Pos2) {
    let mut h = grid(ViewMode::PhotoGrid);
    let first = h.app.session.visible()[0];
    select(&mut h, first);
    let r = h.request("engine.execute", json!({"command": "view.detail"}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.settle(SETTLE);
    let c = h.app.image_rect.expect("photo drawn").center();
    // egui learns there is a touch screen from a first touch elsewhere
    let corner = h.app.canvas_rect.expect("canvas").left_top() + egui::vec2(4.0, 4.0);
    finger_down(&mut h, corner);
    finger_up(&mut h, corner);
    for _ in 0..30 {
        h.step();
    }
    // (on desktop builds that tap also click-zoomed)
    let r = h.request("engine.execute", json!({"command": "view.zoomFit"}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.settle(SETTLE);
    (h, c)
}

fn menu_open(h: &Headless) -> bool {
    widget(h, "context:delete").is_some()
}

/// A long press on the photo in the detail view (Android: shows the original) never opens the
/// context menu.
#[test]
fn long_press_on_the_detail_photo_does_not_open_the_menu() {
    let (mut h, c) = detail();
    finger_down(&mut h, c);
    for _ in 0..90 {
        h.app.synthetic.extend([touch(0, egui::TouchPhase::Move, c), egui::Event::PointerMoved(c)]);
        h.step();
        assert!(!menu_open(&h), "the long press opened the menu");
    }
    finger_up(&mut h, c);
    for _ in 0..5 {
        h.step();
    }
    assert!(!menu_open(&h), "the menu opened after the long press");
    assert_eq!(h.app.ui.view, ViewMode::Detail);
}

/// A quick two-finger tap on the photo opens its context menu; a tap elsewhere closes it again.
#[test]
fn two_finger_tap_on_the_detail_photo_opens_the_menu() {
    let (mut h, c) = detail();
    let zoom = h.app.ui.zoom;
    let (a, b) = (c - egui::vec2(40.0, 0.0), c + egui::vec2(40.0, 0.0));
    h.app.synthetic.extend([touch(0, egui::TouchPhase::Start, a), egui::Event::PointerMoved(a), press(a)]);
    h.step();
    h.app.synthetic.extend([touch(1, egui::TouchPhase::Start, b)]);
    h.step();
    h.app.synthetic.extend([touch(1, egui::TouchPhase::End, b)]);
    h.step();
    h.app.synthetic.extend([touch(0, egui::TouchPhase::End, a), release(a), egui::Event::PointerGone]);
    h.step();
    h.step();
    assert!(menu_open(&h), "the two-finger tap opened the menu");
    let del = widget(&h, "context:delete").expect("menu item");
    assert!(del.left() >= c.x - 60.0 && del.top() >= c.y - 60.0, "the menu opens where the fingers were ({del:?}, {c:?})");
    assert_eq!(h.app.ui.zoom, zoom, "the tap did not zoom {:?}", h.app.ui.zoom);
    // a one-finger tap on the photo (outside the menu) closes it
    let off = h.app.canvas_rect.expect("canvas").left_top() + egui::vec2(30.0, 30.0);
    finger_down(&mut h, off);
    finger_up(&mut h, off);
    h.step();
    assert!(!menu_open(&h), "a tap elsewhere closes the menu");
}

/// A right-click on the photo still opens the menu.
#[test]
fn right_click_on_the_detail_photo_opens_the_menu() {
    let (mut h, c) = detail();
    let r = h.request("ui.click", json!({"x": c.x, "y": c.y, "button": "right"}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.step();
    assert!(menu_open(&h), "right-click opened the menu");
}
