//! An album's Add Photos from Folder… (its "+" and its context menu, `album.importFolder`): the
//! folder opens in the import review with the album chosen, and confirming imports the new photos
//! into All Photos and files them, and the folder's photos the library already had, in the album.
//! A host with its own folder picker (Android) gets the request instead.

use std::time::Duration;

use lightcraft_catalog::AlbumId;
use serde_json::json;

use crate::headless::Headless;
use crate::state::Dialog;
use crate::{LightcraftApp, Services};

const T: Duration = Duration::from_secs(10);
const SETTLE: Duration = Duration::from_secs(120);

fn folder(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("lc-ui-album-import-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for i in 0..3u8 {
        let data: Vec<[u8; 4]> = (0..40 * 30).map(|k| [(k % 40 * 6) as u8, i * 70, 9, 255]).collect();
        let img = lightcraft_raster::Rgba8 { width: 40, height: 30, data };
        let png = lightcraft_codecs::encode_png(&lightcraft_codecs::EncodeImage::rgba8(&img), &lightcraft_codecs::EncodeMeta::default()).unwrap();
        std::fs::write(dir.join(format!("img{i}.png")), png).unwrap();
    }
    dir
}

fn widget(h: &Headless, id: &str) -> Option<egui::Rect> {
    h.app.widgets.iter().find(|(w, _)| w == id).map(|(_, r)| *r)
}

fn app(services: Services) -> Headless {
    let mut app = LightcraftApp::new(lightcraft_engine::Session::new().with_fs(), services);
    app.ui.view = crate::state::ViewMode::PhotoGrid;
    let mut h = Headless::new(app, [1300.0, 900.0], 1.0);
    // My Photos (the albums) on the left
    let r = h.request("engine.execute", json!({"command": "view.leftPanel"}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.settle(SETTLE);
    h
}

fn create_album(h: &mut Headless, name: &str) -> u64 {
    let r = h.request("engine.execute", json!({"command": "album.create", "params": {"name": name}}), T);
    assert_eq!(r["ok"], true, "{r}");
    r["result"]["id"].as_u64().unwrap()
}

/// Desktop: the album's "+" picks a folder and opens the import review with the album chosen;
/// importing adds the two new photos to All Photos and all three (one was in the library) to it.
#[test]
fn the_album_plus_imports_a_folder_into_the_album() {
    let dir = folder("plus");
    let picked = dir.to_string_lossy().to_string();
    let mut h = app(Services { png: None, pick_folder: Some(Box::new(move || Some(picked.clone()))), ..Default::default() });
    // one photo of the folder is in the library already
    let r = h.request("engine.execute", json!({"command": "library.import", "params": {"paths": [dir.join("img0.png").to_string_lossy()]}}), T);
    let existing = r["result"]["imported"][0].as_u64().unwrap();
    let album = create_album(&mut h, "Trip");
    h.settle(SETTLE);
    let plus = format!("icon:albumImportFolder:{album}");
    let ids: Vec<&String> = h.app.widgets.iter().map(|(w, _)| w).filter(|w| w.starts_with("source:") || w.starts_with("sidebar")).collect();
    assert!(widget(&h, &format!("source:album:{album}")).is_some(), "the album row {ids:?}");
    assert!(widget(&h, &plus).is_none(), "no + on an album that is neither selected nor hovered");
    // selected (a tap on a tablet), the album shows its +
    let r = h.request("engine.execute", json!({"command": "library.source", "params": {"kind": "album", "id": album}}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.settle(SETTLE);
    let r = h.request("ui.clickWidget", json!({"id": plus}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.step_until(SETTLE, |h| matches!(h.app.ui.dialog, Some(Dialog::Import { .. })));
    let Some(Dialog::Import { opts }) = &h.app.ui.dialog else { panic!("no import review: {:?}", h.app.ui.dialog) };
    assert_eq!(opts.album, Some(album), "the review imports into the album");
    assert_eq!(opts.candidates.len(), 3);
    assert_eq!(opts.selected_paths().len(), 2, "the photo the library has is not imported again");
    assert_eq!(opts.existing_photos(&h.app.session.catalog), vec![existing]);
    let undo0 = h.app.session.undo.len();
    let r = h.request("ui.dialog.confirm", json!({}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.settle(SETTLE);
    h.step_until(SETTLE, |h| h.app.import.is_none());
    assert_eq!(h.app.session.catalog.len(), 3, "the new photos are in All Photos");
    let members = h.app.session.catalog.album(AlbumId(album)).unwrap().photos.clone();
    assert_eq!(members.len(), 3, "new and existing photos are in the album: {members:?}");
    assert!(members.iter().any(|p| p.0 == existing));
    assert_eq!(h.app.session.undo.len(), undo0 + 1, "one undo step");

    // again: everything is in the library; the review only files them in another album
    let other = create_album(&mut h, "Best of");
    let r = h.request("engine.execute", json!({"command": "album.importFolder", "params": {"id": other}}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.step_until(SETTLE, |h| matches!(h.app.ui.dialog, Some(Dialog::Import { .. })));
    let r = h.request("ui.dialog.confirm", json!({}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.settle(SETTLE);
    assert_eq!(h.app.session.catalog.len(), 3);
    assert_eq!(h.app.session.catalog.album(AlbumId(other)).unwrap().photos.len(), 3);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The album row's context menu (right-click; a long press on a tablet) has the item for a
/// regular album, not for a smart album; it is refused for smart albums by the command too.
#[test]
fn the_album_menu_offers_add_photos_from_folder() {
    let mut h = app(Services { png: None, pick_folder: Some(Box::new(|| None)), ..Default::default() });
    let album = create_album(&mut h, "Trip");
    let r = h.request("engine.execute", json!({"command": "album.createSmart", "params": {"name": "Smart", "rules": {"rating": 3}}}), T);
    let smart = r["result"]["id"].as_u64().unwrap();
    h.settle(SETTLE);
    let row = widget(&h, &format!("source:album:{album}")).expect("album row");
    // hovering shows the +
    let r = h.request("ui.move", json!({"x": row.center().x, "y": row.center().y}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.step();
    h.step();
    assert!(widget(&h, &format!("icon:albumImportFolder:{album}")).is_some(), "+ on hover");
    let r = h.request("ui.click", json!({"x": row.center().x, "y": row.center().y, "button": "right"}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.step();
    h.step();
    assert!(widget(&h, &format!("context:albumImportFolder:{album}")).is_some(), "the menu item");
    h.app.synthetic.push(egui::Event::Key {
        key: egui::Key::Escape,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    });
    h.step();
    h.step();
    let srow = widget(&h, &format!("source:album:{smart}")).expect("smart album row");
    h.request("ui.click", json!({"x": srow.center().x, "y": srow.center().y, "button": "right"}), T);
    h.step();
    h.step();
    assert!(widget(&h, &format!("context:albumImportFolder:{smart}")).is_none(), "not for a smart album");
    assert!(widget(&h, &format!("icon:albumImportFolder:{smart}")).is_none(), "no + on a smart album");
    let r = h.request("engine.execute", json!({"command": "album.importFolder", "params": {"id": smart}}), T);
    assert_eq!(r["ok"], false, "{r}");
}

/// On a tablet the album menu opens with a long press on the row (egui counts a long touch as a
/// right-click), so Add Photos from Folder… is reachable by touch.
#[test]
fn a_long_press_on_an_album_opens_its_menu() {
    let mut h = app(Services { png: None, host_folder_import: true, ..Default::default() });
    let album = create_album(&mut h, "Trip");
    h.settle(SETTLE);
    let p = widget(&h, &format!("source:album:{album}")).expect("album row").left_center() + egui::vec2(60.0, 0.0);
    let touch = |phase| egui::Event::Touch { device_id: egui::TouchDeviceId(0), id: egui::TouchId(0), phase, pos: p, force: None };
    let button = |pressed| egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
    h.app.synthetic.extend([touch(egui::TouchPhase::Start), egui::Event::PointerMoved(p), button(true)]);
    h.step();
    let mut opened = false;
    for _ in 0..120 {
        h.app.synthetic.extend([touch(egui::TouchPhase::Move), egui::Event::PointerMoved(p)]);
        h.step();
        opened |= widget(&h, &format!("context:albumImportFolder:{album}")).is_some();
    }
    h.app.synthetic.extend([touch(egui::TouchPhase::End), button(false), egui::Event::PointerGone]);
    h.step();
    h.step();
    let item = widget(&h, &format!("context:albumImportFolder:{album}"));
    assert!(opened || item.is_some(), "the long press opened the album menu");
    // tapping the item asks the host for a folder
    let item = item.expect("the menu stays open after the finger lifts");
    let q = item.center();
    let touch = |phase| egui::Event::Touch { device_id: egui::TouchDeviceId(0), id: egui::TouchId(1), phase, pos: q, force: None };
    let button = |pressed| egui::Event::PointerButton { pos: q, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
    h.app.synthetic.extend([touch(egui::TouchPhase::Start), egui::Event::PointerMoved(q), button(true)]);
    h.step();
    h.app.synthetic.extend([touch(egui::TouchPhase::End), button(false), egui::Event::PointerGone]);
    h.step();
    h.step();
    assert_eq!(h.app.ui.request_folder_import, Some(album));
}

/// Android: without a folder dialog of ours the request is left for the host (which opens the
/// system picker and imports into the album); a platform with neither offers nothing.
#[test]
fn a_host_with_its_own_picker_gets_the_request() {
    let mut h = app(Services { png: None, host_folder_import: true, ..Default::default() });
    let album = create_album(&mut h, "Trip");
    h.settle(SETTLE);
    let r = h.request("engine.execute", json!({"command": "library.source", "params": {"kind": "album", "id": album}}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.settle(SETTLE);
    let r = h.request("ui.clickWidget", json!({"id": format!("icon:albumImportFolder:{album}")}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.step();
    assert_eq!(h.app.ui.request_folder_import, Some(album), "the host is asked");
    assert!(h.app.scan.is_none() && h.app.ui.dialog.is_none(), "no desktop review");

    let mut h = app(Services { png: None, ..Default::default() });
    let album = create_album(&mut h, "Trip");
    let r = h.request("engine.execute", json!({"command": "library.source", "params": {"kind": "album", "id": album}}), T);
    assert_eq!(r["ok"], true, "{r}");
    h.settle(SETTLE);
    assert!(widget(&h, &format!("icon:albumImportFolder:{album}")).is_none(), "no folder picker, no +");
}
