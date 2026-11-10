package com.lightcraft.android;

import android.content.Context;
import android.os.SystemClock;
import android.text.InputType;
import android.view.KeyCharacterMap;
import android.view.KeyEvent;
import android.view.View;
import android.view.inputmethod.BaseInputConnection;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputConnection;

/**
 * The soft keyboard's target. NativeActivity's own view has no InputConnection, so keyboards
 * fall back to a dummy one that drops {@code deleteSurroundingText} (how Gboard and others
 * usually delete) and holds composing text back. This view gives them a connection that turns
 * everything into key events, which reach Rust through NativeActivity's input queue like a
 * hardware keyboard's.
 */
final class ImeView extends View {
    ImeView(Context context) {
        super(context);
        setFocusable(true);
        setFocusableInTouchMode(true);
    }

    @Override public boolean onCheckIsTextEditor() { return true; }

    /** winit shows/hides the keyboard for the decor view; let this view serve it. */
    @Override public boolean checkInputConnectionProxy(View view) { return true; }

    @Override public InputConnection onCreateInputConnection(EditorInfo out) {
        // No suggestions: most keyboards then commit each key instead of composing whole words
        // (the text lives in Rust, the keyboard can't see it). Composing ones still work, see
        // setComposingText. Not a password variation: that would force a Latin layout.
        out.inputType = InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS;
        out.imeOptions = EditorInfo.IME_FLAG_NO_EXTRACT_UI | EditorInfo.IME_FLAG_NO_FULLSCREEN | EditorInfo.IME_ACTION_DONE;
        out.initialSelStart = 0;
        out.initialSelEnd = 0;
        return new Connection(this);
    }

    private static final class Connection extends BaseInputConnection {
        /** Code points of composing text already sent as keys (replaced on the next update). */
        private int composing = 0;

        Connection(View view) { super(view, false); }

        private void key(int code) {
            long now = SystemClock.uptimeMillis();
            int flags = KeyEvent.FLAG_SOFT_KEYBOARD | KeyEvent.FLAG_KEEP_TOUCH_MODE;
            sendKeyEvent(new KeyEvent(now, now, KeyEvent.ACTION_DOWN, code, 0, 0, KeyCharacterMap.VIRTUAL_KEYBOARD, 0, flags));
            sendKeyEvent(new KeyEvent(now, now, KeyEvent.ACTION_UP, code, 0, 0, KeyCharacterMap.VIRTUAL_KEYBOARD, 0, flags));
        }

        private void keys(int code, int count) {
            for (int i = 0; i < Math.max(0, Math.min(count, 4096)); i++) key(code);
        }

        private void dropComposing() {
            keys(KeyEvent.KEYCODE_DEL, composing);
            composing = 0;
        }

        @Override public boolean deleteSurroundingText(int before, int after) {
            keys(KeyEvent.KEYCODE_DEL, before);
            keys(KeyEvent.KEYCODE_FORWARD_DEL, after);
            return true;
        }

        @Override public boolean deleteSurroundingTextInCodePoints(int before, int after) {
            return deleteSurroundingText(before, after);
        }

        @Override public boolean commitText(CharSequence text, int newCursorPosition) {
            dropComposing();
            // dummy mode: the base class sends the text as key events and clears its buffer
            return super.commitText(text, newCursorPosition);
        }

        @Override public boolean setComposingText(CharSequence text, int newCursorPosition) {
            dropComposing();
            super.commitText(text, newCursorPosition);
            composing = text == null ? 0 : Character.codePointCount(text, 0, text.length());
            return true;
        }

        @Override public boolean finishComposingText() {
            composing = 0;
            return true;
        }

        @Override public boolean performEditorAction(int action) {
            composing = 0;
            key(KeyEvent.KEYCODE_ENTER);
            return true;
        }
    }
}
