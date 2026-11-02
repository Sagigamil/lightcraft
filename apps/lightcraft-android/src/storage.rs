//! SAF document identity policy, independent of Android's VM for testing.
#![forbid(unsafe_code)]

pub fn validate_uri(uri: &str) -> Result<(), String> {
    let authority = uri.strip_prefix("content://").and_then(|s| s.split('/').next()).unwrap_or_default();
    if authority.is_empty() || uri.chars().any(char::is_control) {
        return Err("A valid Android content URI is required".into());
    }
    Ok(())
}

pub fn export_name(path: &str) -> Result<&str, String> {
    let name = path.rsplit(['/', '\\']).next().unwrap_or_default();
    if name.is_empty() || name == "." || name == ".." || name.chars().any(char::is_control) {
        return Err("Invalid export filename".into());
    }
    Ok(name)
}

/// Apply the destination returned by the Android folder picker.
pub fn set_export_destination(app: &mut lightcraft_ui_egui::LightcraftApp, destination: &str) {
    // The shared dialog and Export with Previous read last_export. Android's
    // persisted destination is authoritative even when no dialog is open.
    let previous = app.session.last_export.get_or_insert_with(|| serde_json::json!({}));
    if !previous.is_object() {
        *previous = serde_json::json!({});
    }
    previous["dir"] = serde_json::json!(destination);
    if let Some(lightcraft_ui_egui::state::Dialog::Export { dir, .. }) = &mut app.ui.dialog {
        *dir = destination.into();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_content_uri_without_interpreting_document_id_as_path() {
        assert!(validate_uri("content://com.android.externalstorage.documents/tree/1234%3ADCIM/document/1234%3ADCIM%2Fphoto.NEF").is_ok());
        assert!(validate_uri("/sdcard/DCIM/photo.NEF").is_err());
        assert!(validate_uri("file:///private/photo.NEF").is_err());
        assert!(validate_uri("content://").is_err());
    }

    #[test]
    fn destination_name_cannot_escape_or_overwrite_original_path() {
        assert_eq!(export_name("/virtual/export/edited.jpg").unwrap(), "edited.jpg");
        assert!(export_name("../..").is_err());
        assert!(export_name("").is_err());
        assert!(export_name("photo\0.jpg").is_err());
    }

    #[test]
    fn toolbar_destination_is_used_when_export_dialog_opens_later() {
        let mut app = lightcraft_ui_egui::LightcraftApp::new(lightcraft_engine::Session::new(), Default::default());
        app.session.last_export = Some(serde_json::json!({"dir":"old-folder", "quality":73}));
        set_export_destination(&mut app, "content://provider/tree/exports");
        app.run("dialog.export", serde_json::json!({})).unwrap();
        let Some(lightcraft_ui_egui::state::Dialog::Export { dir, .. }) = &app.ui.dialog else { panic!("export dialog missing") };
        assert_eq!(dir, "content://provider/tree/exports");
        assert_eq!(app.session.last_export.as_ref().unwrap()["quality"], 73);
    }

    #[test]
    fn destination_updates_open_dialog_and_subsequent_exports() {
        let mut app = lightcraft_ui_egui::LightcraftApp::new(lightcraft_engine::Session::new(), Default::default());
        app.run("dialog.export", serde_json::json!({})).unwrap();
        set_export_destination(&mut app, "content://provider/tree/new");
        let Some(lightcraft_ui_egui::state::Dialog::Export { dir, .. }) = &app.ui.dialog else { panic!("export dialog missing") };
        assert_eq!(dir, "content://provider/tree/new");
        app.ui.dialog = None;
        app.run("dialog.export", serde_json::json!({})).unwrap();
        let Some(lightcraft_ui_egui::state::Dialog::Export { dir, .. }) = &app.ui.dialog else { panic!("export dialog missing") };
        assert_eq!(dir, "content://provider/tree/new");
    }
}
