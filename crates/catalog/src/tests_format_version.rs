//! Issue #102: catalog format versions. Older formats load and are upgraded; a newer format is
//! refused with a clear error and left untouched (never read in part, truncated or set aside).

use crate::journal::{LOG, SNAPSHOT, VERSION, encode_record};
use crate::*;

fn photo(c: &mut Catalog) -> Op {
    let id = c.alloc_photo_id();
    Op::AddPhoto { photo: Box::new(Photo::new(id, Source::File { path: format!("/p/{}.jpg", id.0) }, "p.jpg", "JPEG", 4, 3, "2026-01-01")) }
}

/// A catalog with two photos and the log line for a third op on top of it.
fn legacy_parts() -> (Catalog, String, Catalog) {
    let mut c = Catalog::new();
    for _ in 0..2 {
        let op = photo(&mut c);
        c.apply(op).unwrap();
    }
    let base = c.clone();
    let id = c.photos().next().unwrap().id;
    let op = Op::SetRating { id, rating: 4 };
    c.apply(op.clone()).unwrap();
    (base, format!("{}\n", encode_record(3, &op)), c)
}

fn snapshot_version(m: &MemStore) -> u64 {
    let v: serde_json::Value = serde_json::from_slice(&m.get(SNAPSHOT).unwrap()).unwrap();
    v["version"].as_u64().unwrap()
}

#[test]
fn v1_library_loads_and_is_upgraded() {
    let (base, log, full) = legacy_parts();
    let m = MemStore::new();
    m.set(SNAPSHOT, format!("{{\"format\":\"lightcraft-catalog\",\"version\":1,\"seq\":2,\"catalog\":{}}}\n", base.to_snapshot()).into_bytes());
    m.set(LOG, log.into_bytes());
    let (mut j, c, r) = Journal::open(Box::new(m.clone())).unwrap();
    assert_eq!(c.to_snapshot(), full.to_snapshot());
    assert_eq!((r.replayed, r.upgraded_from), (1, Some(1)));
    // rewritten in this format before anything is appended
    assert_eq!(snapshot_version(&m), u64::from(VERSION));
    assert!(m.get(LOG).unwrap().is_empty());
    assert_eq!(j.seq(), 3);
    j.append(&[Op::SetLabelName { label: ColorLabel::ALL[0], name: Some("x".into()) }]).unwrap();
    drop(j);
    let (_, _, r) = Journal::open(Box::new(m)).unwrap();
    assert_eq!((r.replayed, r.upgraded_from), (1, None));
}

/// A library saved before albums could be ordered by hand opens as it was and is rewritten in the current format.
#[test]
fn v2_library_loads_and_is_upgraded() {
    let (base, log, full) = legacy_parts();
    let m = MemStore::new();
    m.set(SNAPSHOT, format!("{{\"format\":\"lightcraft-catalog\",\"version\":2,\"seq\":2,\"catalog\":{}}}\n", base.to_snapshot()).into_bytes());
    m.set(LOG, log.into_bytes());
    let (_, c, r) = Journal::open(Box::new(m.clone())).unwrap();
    assert_eq!(c.to_snapshot(), full.to_snapshot());
    assert_eq!((r.replayed, r.upgraded_from), (1, Some(2)));
    assert_eq!(snapshot_version(&m), u64::from(VERSION));
}

/// A library saved before keywords could be listed on their own opens as it was and is rewritten
/// in the current format.
#[test]
fn v3_library_loads_and_is_upgraded() {
    let (base, log, full) = legacy_parts();
    let m = MemStore::new();
    m.set(SNAPSHOT, format!("{{\"format\":\"lightcraft-catalog\",\"version\":3,\"seq\":2,\"catalog\":{}}}\n", base.to_snapshot()).into_bytes());
    m.set(LOG, log.into_bytes());
    let (_, c, r) = Journal::open(Box::new(m.clone())).unwrap();
    assert_eq!(c.to_snapshot(), full.to_snapshot());
    assert_eq!((r.replayed, r.upgraded_from), (1, Some(3)));
    assert_eq!(snapshot_version(&m), u64::from(VERSION));
}

/// A v6 library (before preview-only photos recorded which version decided it, #662) opens with
/// an empty stamp on its preview-only photos, so they are rechecked once, and is rewritten in the
/// current format.
#[test]
fn v6_preview_only_photo_loads_unstamped_and_is_upgraded() {
    let mut c = Catalog::new();
    let id = c.alloc_photo_id();
    let mut p = Photo::new(id, Source::File { path: "/p/a.cr3".into() }, "a.cr3", "CR3", 160, 120, "2026-01-01");
    p.kind = MediaKind::Raw;
    p.preview_only = Some("Cr3 files are not decoded yet".into());
    c.apply(Op::AddPhoto { photo: Box::new(p) }).unwrap();
    let snap = c.to_snapshot();
    assert!(!snap.contains("preview_checked"), "an empty stamp isn't written");
    let m = MemStore::new();
    m.set(SNAPSHOT, format!("{{\"format\":\"lightcraft-catalog\",\"version\":6,\"seq\":1,\"catalog\":{snap}}}\n").into_bytes());
    let (_, loaded, r) = Journal::open(Box::new(m.clone())).unwrap();
    let p = loaded.photo(id).unwrap();
    assert_eq!(p.preview_only.as_deref(), Some("Cr3 files are not decoded yet"));
    assert_eq!(p.preview_checked, "");
    assert_eq!(r.upgraded_from, Some(6));
    assert_eq!(snapshot_version(&m), u64::from(VERSION));
}

#[test]
fn versionless_log_only_library_is_upgraded() {
    let mut c = Catalog::new();
    let op = photo(&mut c);
    c.apply(op.clone()).unwrap();
    let m = MemStore::new();
    m.set(LOG, format!("{}\n", encode_record(1, &op)).into_bytes());
    let (_, loaded, r) = Journal::open(Box::new(m.clone())).unwrap();
    assert_eq!(loaded.to_snapshot(), c.to_snapshot());
    assert_eq!(r.upgraded_from, Some(0));
    assert_eq!(snapshot_version(&m), u64::from(VERSION));
}

/// A new library gets a snapshot in this format at once, so an older build never sees a bare log
/// it would read in part.
#[test]
fn new_library_is_written_in_this_format() {
    let m = MemStore::new();
    let (_, _, r) = Journal::open(Box::new(m.clone())).unwrap();
    assert!(r.created);
    assert_eq!(r.upgraded_from, None);
    assert_eq!(snapshot_version(&m), u64::from(VERSION));
}

fn assert_refused_untouched(m: &MemStore) {
    let before = m.files.lock().unwrap().clone();
    let e = Journal::open(Box::new(m.clone())).err().expect("refused");
    assert!(matches!(e, CatalogError::Newer(_)), "{e:?}");
    let msg = e.to_string();
    assert!(msg.contains("newer version of LightCraft") && msg.contains("left unchanged"), "{msg}");
    assert_eq!(*m.files.lock().unwrap(), before, "nothing modified, nothing renamed");
}

#[test]
fn newer_snapshot_is_refused_untouched() {
    let (base, log, _) = legacy_parts();
    let m = MemStore::new();
    m.set(
        SNAPSHOT,
        format!("{{\"format\":\"lightcraft-catalog\",\"version\":{},\"seq\":2,\"catalog\":{}}}\n", VERSION + 1, base.to_snapshot()).into_bytes(),
    );
    m.set(LOG, log.into_bytes());
    assert_refused_untouched(&m);
    // even when its catalog doesn't parse as this version's
    let m = MemStore::new();
    m.set(
        SNAPSHOT,
        format!("{{\"format\":\"lightcraft-catalog\",\"version\":{},\"seq\":2,\"catalog\":{{\"photos\":7}}}}\n", VERSION + 1).into_bytes(),
    );
    assert_refused_untouched(&m);
    let e = Journal::open(Box::new(m)).err().unwrap().to_string();
    assert!(e.contains(&format!("v{}", VERSION + 1)), "{e}");
}

/// A log record with a valid CRC whose op this version doesn't know: refused, not taken for a
/// torn tail (truncated) or damage (renamed) — wherever it is in the log.
#[test]
fn unknown_op_in_the_log_is_refused_untouched() {
    let (base, log, _) = legacy_parts();
    let future = |seq: u64| {
        let body = r#"{"op":"fromTheFuture","id":1,"what":"something new"}"#;
        format!("{{\"seq\":{seq},\"crc\":{},\"op\":{body}}}\n", crc32fast::hash(body.as_bytes()))
    };
    let snap = format!("{{\"format\":\"lightcraft-catalog\",\"version\":{VERSION},\"seq\":2,\"catalog\":{}}}\n", base.to_snapshot());
    let id = base.photos().next().unwrap().id;
    let after = format!("{}\n", encode_record(5, &Op::SetRating { id, rating: 1 }));
    for log in [format!("{log}{}", future(4)), format!("{log}{}{after}", future(4)), format!("{}{log}", future(3))] {
        let m = MemStore::new();
        m.set(SNAPSHOT, snap.clone().into_bytes());
        m.set(LOG, log.into_bytes());
        assert_refused_untouched(&m);
    }
    // a known op with a field of an unknown shape is "newer" too
    let body = r#"{"op":"setRating","id":1,"rating":"five stars"}"#;
    let m = MemStore::new();
    m.set(SNAPSHOT, snap.into_bytes());
    m.set(LOG, format!("{log}{{\"seq\":4,\"crc\":{},\"op\":{body}}}\n", crc32fast::hash(body.as_bytes())).into_bytes());
    assert_refused_untouched(&m);
}

/// A CRC mismatch is still damage / a torn tail (not "newer").
#[test]
fn bad_crc_is_still_a_torn_tail() {
    let (base, log, _) = legacy_parts();
    let m = MemStore::new();
    m.set(
        SNAPSHOT,
        format!("{{\"format\":\"lightcraft-catalog\",\"version\":{VERSION},\"seq\":2,\"catalog\":{}}}\n", base.to_snapshot()).into_bytes(),
    );
    let body = r#"{"op":"fromTheFuture"}"#;
    m.set(LOG, format!("{log}{{\"seq\":4,\"crc\":1,\"op\":{body}}}\n").into_bytes());
    let (_, _, r) = Journal::open(Box::new(m)).unwrap();
    assert!(r.torn_bytes > 0 && r.damaged.is_none(), "{r:?}");
}

/// Guard for the version rule (see `journal` → *Format versions*): adding an [`Op`] variant breaks
/// this match. When it does, bump `journal::VERSION`, add a row to the format table, and add the
/// variant here under the new version. A version that only adds serialized fields raises
/// `NEWEST_FIELDS` instead.
#[test]
fn op_variants_are_versioned() {
    fn since(op: &Op) -> u32 {
        match op {
            Op::AddPhoto { .. }
            | Op::RemovePhoto { .. }
            | Op::SetRating { .. }
            | Op::SetFlag { .. }
            | Op::SetLabel { .. }
            | Op::SetDevelop { .. }
            | Op::SetMeta { .. }
            | Op::SetDeleted { .. }
            | Op::SetLocal { .. }
            | Op::SetVersions { .. }
            | Op::SetHistory { .. }
            | Op::PushHistory { .. }
            | Op::AddAlbum { .. }
            | Op::RemoveAlbum { .. }
            | Op::RenameAlbum { .. }
            | Op::MoveAlbum { .. }
            | Op::SetAlbumPhotos { .. }
            | Op::SetAlbumCover { .. }
            | Op::SetAlbumRules { .. }
            | Op::AddStack { .. }
            | Op::RemoveStack { .. }
            | Op::SetStack { .. }
            | Op::SetCaptured { .. }
            | Op::SetAnalysis { .. }
            | Op::SetFile { .. }
            | Op::Relink { .. }
            | Op::SetContent { .. }
            | Op::SetLabelName { .. }
            | Op::Batch { .. } => 1,
            Op::SetBrowsed { .. } => 2,
            Op::SetAlbumOrder { .. } => 3,
            Op::SetEmbeddedLens { .. } => 4,
            Op::SetKeyword { .. } => 5,
            Op::SetFolderRecord { .. } => 6,
            Op::SetAsShotWb { .. } => 7,
            Op::SetHistoryCursor { .. } => 8,
        }
    }
    /// v13: Lens Blur in develop settings and mask adjustments (v12: `Catalog.grids` and `SegMask.grid`; v11: `EmbeddedLens.source`;
    /// v10: `Mask.on_photo`; v9: `Photo.preview_checked` and `Op::SetContent.preview_checked`).
    const NEWEST_FIELDS: u32 = 13;
    let newest = since(&Op::SetHistoryCursor { id: PhotoId(1), cursor: None });
    assert_eq!(newest.max(NEWEST_FIELDS), VERSION, "the newest op's or fields' version must be the current format version");
}

/// Every field the v2 build did not know (format 3-13) survives a snapshot: album order, the
/// keyword list, folder records, the History cursor, the As Shot stamp, the preview stamp, embedded
/// lens data, the WB scale, masks on the photo with Lens Blur, shared segmentation grids and Lens
/// Blur settings. Written in v13, read back equal.
#[test]
fn a_v13_library_keeps_every_format_field() {
    use crate::folders::FolderRecord;
    let mut c = Catalog::new();
    let op = photo(&mut c);
    c.apply(op).unwrap();
    let id = c.photos().next().unwrap().id;
    let aid = c.alloc_album_id();
    c.apply(Op::AddAlbum { album: Album::new(aid, "Trip") }).unwrap();
    c.apply(Op::SetAlbumOrder { id: aid, order: Some(4) }).unwrap();
    let info: crate::keywords::KeywordInfo = serde_json::from_value(serde_json::json!({"synonyms": ["italia"]})).unwrap();
    c.apply(Op::SetKeyword { path: "travel|italy".into(), info: Some(info) }).unwrap();
    c.apply(Op::SetFolderRecord { folder: "/p".into(), record: Some(FolderRecord { label: Some(ColorLabel::Red) }) }).unwrap();
    c.apply(Op::SetHistoryCursor { id, cursor: Some(0) }).unwrap();
    c.apply(Op::SetAsShotWb { id, wb: Some((5000.0, 3.0)) }).unwrap();
    c.apply(Op::SetEmbeddedLens {
        id,
        lens: Some(Box::new(lightcraft_develop::EmbeddedLens { source: lightcraft_develop::LensSource::Database, ..Default::default() })),
    })
    .unwrap();
    let mut d = DevelopSettings::for_raw(5000.0, 3.0);
    d.wb.scale = lightcraft_develop::WbScale::Kelvin;
    d.lens_blur.enabled = true;
    d.lens_blur.amount = 40.0;
    d.lens_blur.transition = 70.0;
    d.masks.push(lightcraft_develop::Mask {
        id: 1,
        on_photo: false,
        adjust: lightcraft_develop::LocalAdjustments { lens_blur: 20.0, ..Default::default() },
        components: vec![lightcraft_develop::MaskComponent {
            name: None,
            op: Default::default(),
            invert: false,
            shape: lightcraft_develop::MaskShape::Object {
                hint: Vec::new(),
                exclude: Vec::new(),
                seg: Some(lightcraft_develop::SegMask::from_logits(4, &[1.0; 16])),
                detail: Vec::new(),
                edge: 0.0,
            },
        }],
        ..Default::default()
    });
    c.apply(Op::SetDevelop { id, settings: Arc::new(d), label: "Lens Blur".into(), edited: None }).unwrap();
    c.apply(Op::SetContent {
        id,
        width: 4,
        height: 3,
        file_size: 10,
        content_hash: None,
        preview_only: Some("preview".into()),
        preview_checked: "9.9.9".into(),
    })
    .unwrap();

    let m = MemStore::new();
    let (mut j, _, _) = Journal::open(Box::new(m.clone())).unwrap();
    j.snapshot(&c).unwrap();
    drop(j);
    assert_eq!(snapshot_version(&m), u64::from(VERSION));
    // the shared grid table holds the one grid once, and no mask repeats it inline
    let snap: serde_json::Value = serde_json::from_slice(&m.get(SNAPSHOT).unwrap()).unwrap();
    assert_eq!(snap["catalog"]["grids"].as_array().map(Vec::len), Some(1));

    let (_, back, r) = Journal::open(Box::new(m)).unwrap();
    assert_eq!(r.upgraded_from, None);
    assert_eq!(back.to_snapshot(), c.to_snapshot(), "every v13 field round-trips");
}
