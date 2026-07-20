# Running on Linux

Two things stand between a freshly plugged-in dongle and a browser that can talk to it.
Neither is something the application can fix from inside the browser; both are one-time
system configuration.

## The kernel driver claims the device

The kernel ships a DVB television driver, `dvb_usb_rtl28xxu`, that binds any RTL2832U device
the moment it is plugged in. It exposes the dongle as a television tuner, which is not what a
software radio wants — and while it holds the single USB interface, nothing else can claim
it.

Native tools such as `rtl_test` detach it automatically. **WebUSB cannot.** There is no API
for detaching a kernel driver, Chromium's automatic-detach feature is disabled by default on
Linux, and even enabled it only detaches an allowlist of three drivers that does not include
this one. So the driver has to be prevented from binding in the first place:

```sh
echo 'blacklist dvb_usb_rtl28xxu' | sudo tee /etc/modprobe.d/blacklist-rtlsdr.conf
sudo modprobe -r dvb_usb_rtl28xxu   # unload it now; it stays gone after a reboot
```

Then unplug and replug the dongle. Confirm nothing has it:

```sh
ls -l /sys/bus/usb/devices/*/*/driver 2>/dev/null | grep -i rtl   # should print nothing
```

## Your user needs access to the device node

USB device nodes are owned by root and are not world-writable. The browser runs as you, so
you need read and write on `/dev/bus/usb/…` for the dongle.

The robust way is a udev rule that tags the device for the logged-in user rather than
relying on a group that not every distribution defines:

```
# /etc/udev/rules.d/60-rtl-sdr.rules
SUBSYSTEMS=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", TAG+="uaccess"
```

```sh
sudo udevadm control --reload-rules
# replug the dongle
```

`TAG+="uaccess"` hands the device to whoever is logged in at the local seat, through
systemd-logind, with no group membership needed.

## The shortcut

Your distribution's `rtl-sdr` package installs both the blacklist and a `uaccess` udev rule.
On Arch, `pacman -S rtl-sdr` is enough; on Debian and derivatives, `apt install rtl-sdr`.
You do not need the package's tools — only the two configuration files it drops.

## Telling the two failures apart

The application separates them, because the fixes are completely different:

- **"Access denied"** — the udev permissions are missing. Your user cannot open the device
  node. Fix the udev rule.
- **"Could not claim the interface"** — the kernel driver still holds it. Fix the blacklist,
  and make sure you replugged after unloading the module.

## Browser packaging

A sandboxed browser package can block USB access even with the permissions right. A Flatpak
Chromium already carries the USB permission; a Snap Chromium historically needed
`snap connect chromium:raw-usb`. A distribution package or the official Chrome build avoids
the question. If enumeration works but the device never appears in the chooser, check
`chrome://device-log`.

## macOS and Windows

macOS needs none of this — there is no in-kernel DVB driver to detach, and access is granted
by default.

Windows needs the WinUSB driver bound to the device, which Zadig installs. Chromium
recognises WinUSB specifically; libusbK and libusb-win32 will not work. A Windows update can
silently revert the binding, in which case re-run Zadig.
