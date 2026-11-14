//! Flags, ratings and colour labels reach the same-folder files with the same base name and
//! another extension (x.jpg ↔ x.raw) when `Session::variant_sync` is on, and photo edits are
//! never shared.

use serde_json::json;

use crate::Session;
use lightcraft_catalog::{ColorLabel, Flag, MediaKind, Op, Photo, PhotoId, Source};

fn add(s: &mut Session, id: u64, path: &str, format: &str, kind: MediaKind) -> PhotoId {
    let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
    let mut p = Photo::new(PhotoId(id), Source::File { path: path.to_string() }, name, format, 16, 16, "2026-01-01T00:00:00");
    p.kind = kind;
    s.catalog.apply(Op::AddPhoto { photo: Box::new(p) }).unwrap();
    PhotoId(id)
}

fn photo(s: &Session, id: PhotoId) -> &Photo {
    s.catalog.photo(id).expect("photo")
}

fn select(s: &mut Session, id: u64) {
    s.execute("library.select", &json!({"ids": [id], "active": id})).unwrap();
}

#[test]
fn flag_rating_and_label_reach_same_folder_variants() {
    let mut s = Session::new();
    let jpg = add(&mut s, 1, "/photos/IMG_0001.jpg", "JPEG", MediaKind::Image);
    let raw = add(&mut s, 2, "/photos/IMG_0001.CR3", "CR3", MediaKind::Raw);
    let other = add(&mut s, 3, "/other/IMG_0001.jpg", "JPEG", MediaKind::Image);
    select(&mut s, 1);
    s.execute("photo.flag", &json!({"ids": [1], "flag": "pick"})).unwrap();
    s.execute("photo.rate", &json!({"ids": [1], "rating": 4})).unwrap();
    s.execute("photo.label", &json!({"ids": [1], "label": "red"})).unwrap();
    assert_eq!(photo(&s, jpg).flag, Flag::Pick);
    assert_eq!(photo(&s, raw).flag, Flag::Pick, "the raw sibling is flagged too");
    assert_eq!(photo(&s, raw).rating, 4, "and rated");
    assert_eq!(photo(&s, raw).label, Some(ColorLabel::Red), "and labelled");
    assert_eq!(photo(&s, other).flag, Flag::None, "another folder is not touched");
    assert_eq!(photo(&s, other).rating, 0);
    assert_eq!(photo(&s, other).label, None);
}

#[test]
fn switching_it_off_keeps_them_independent() {
    let mut s = Session::new();
    add(&mut s, 1, "/photos/IMG_0001.jpg", "JPEG", MediaKind::Image);
    add(&mut s, 2, "/photos/IMG_0001.CR3", "CR3", MediaKind::Raw);
    let r = s.execute("library.preferences", &json!({"variantSync": false})).unwrap();
    assert_eq!(r["variantSync"], false);
    assert!(!s.variant_sync);
    select(&mut s, 1);
    s.execute("photo.flag", &json!({"ids": [1], "flag": "reject"})).unwrap();
    assert_eq!(photo(&s, PhotoId(1)).flag, Flag::Reject);
    assert_eq!(photo(&s, PhotoId(2)).flag, Flag::None, "off: only the target changes");
}

#[test]
fn develop_edits_are_not_shared() {
    let mut s = Session::new();
    add(&mut s, 1, "/photos/IMG_0001.jpg", "JPEG", MediaKind::Image);
    add(&mut s, 2, "/photos/IMG_0001.CR3", "CR3", MediaKind::Raw);
    select(&mut s, 1);
    s.execute("develop.set", &json!({"control": "light.exposure", "value": 1.0})).unwrap();
    assert_eq!(photo(&s, PhotoId(1)).develop.light.exposure, 1.0);
    assert_eq!(photo(&s, PhotoId(2)).develop.light.exposure, 0.0, "photo edits stay per file");
}
