using System.Diagnostics;

namespace Zoidberg.utils;

public static class Shell {
    public static void Bash(this string cmd) {
        var escapedArgs = cmd.Replace("\"", "\\\"");

        new Thread(() => Run("/bin/bash", $"-c \"{escapedArgs}\"")).Start();
    }

    public static void Bat(this string cmd) {
        var escapedArgs = cmd.Replace("\"", "\\\"");

        new Thread(() => Run("cmd.exe", $"/c \"{escapedArgs}\"")).Start();
    }

    private static void Run(string filename, string arguments) {
        var process = new Process {
            StartInfo = new ProcessStartInfo {
                FileName = filename,
                Arguments = arguments,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            }
        };

        process.Start();
        process.WaitForExit();
    }
}