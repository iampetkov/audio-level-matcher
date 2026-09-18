#!/bin/bash
# Audio Level Matcher — Installer

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_SRC="$SCRIPT_DIR/AudioLevelMatcher.app"
SUPPORT_DIR="$HOME/Library/Application Support/AudioLevelMatcher"
VENV_DIR="$SUPPORT_DIR/venv"

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

if [ ! -d "$APP_SRC" ]; then
    osascript -e 'display alert "Install failed" message "Could not find AudioLevelMatcher.app next to the installer."'
    exit 1
fi

# Find Python
for candidate in "/opt/homebrew/bin/python3.14" "/opt/homebrew/bin/python3.13" "/opt/homebrew/bin/python3.12" "/opt/homebrew/bin/python3" "/usr/local/bin/python3"; do
    if [ -x "$candidate" ]; then PYTHON="$candidate"; break; fi
done

if [ -z "$PYTHON" ]; then
    osascript -e 'display alert "Python not found" message "Please install Python 3:\n\nbrew install python3"'
    exit 1
fi

echo "Using Python: $PYTHON"

# Install app
echo "Copying to /Applications..."
rm -rf /Applications/AudioLevelMatcher.app
cp -r "$APP_SRC" /Applications/AudioLevelMatcher.app
xattr -cr /Applications/AudioLevelMatcher.app
chmod +x /Applications/AudioLevelMatcher.app/Contents/MacOS/AudioLevelMatcher

# Set up venv
echo "Setting up Python environment..."
mkdir -p "$SUPPORT_DIR"
rm -rf "$VENV_DIR"
"$PYTHON" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install soundfile pyloudnorm numpy -q

if [ $? -ne 0 ]; then
    osascript -e 'display alert "Setup failed" message "Could not install required libraries.\nMake sure you have an internet connection."'
    rm -rf "$VENV_DIR"
    exit 1
fi

# Create desktop launcher as a shell script wrapped in an .app
echo "Creating Desktop launcher..."
LAUNCHER="$HOME/Desktop/AudioLevelMatcher.app"
rm -rf "$LAUNCHER"
mkdir -p "$LAUNCHER/Contents/MacOS"

cat > "$LAUNCHER/Contents/MacOS/AudioLevelMatcher" << 'LAUNCHEOF'
#!/bin/bash
exec /Applications/AudioLevelMatcher.app/Contents/MacOS/AudioLevelMatcher
LAUNCHEOF
chmod +x "$LAUNCHER/Contents/MacOS/AudioLevelMatcher"

cat > "$LAUNCHER/Contents/Info.plist" << 'PLISTEOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>AudioLevelMatcher</string>
    <key>CFBundleDisplayName</key><string>Audio Level Matcher</string>
    <key>CFBundleIdentifier</key><string>com.audiotools.levelMatcher.launcher</string>
    <key>CFBundleVersion</key><string>1.0</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleExecutable</key><string>AudioLevelMatcher</string>
    <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLISTEOF

xattr -cr "$LAUNCHER"

echo "Installation complete!"
osascript -e 'display notification "Installation complete! Click AudioLevelMatcher on your Desktop." with title "Audio Level Matcher"'

# Launch
exec /Applications/AudioLevelMatcher.app/Contents/MacOS/AudioLevelMatcher
