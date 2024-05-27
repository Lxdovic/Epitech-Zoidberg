namespace Zoidberg.utils;

public static class Clipboard {
    public static void Copy(string val) {
        if (OperatingSystem.IsWindows()) $"echo {val} | clip".Bat();
        if (OperatingSystem.IsMacOS()) $"echo \"{val}\" | pbcopy".Bash();
        if (OperatingSystem.IsLinux()) $"echo \"{val}\" | xclip -selection clipboard".Bash();
    }
}