using System.Numerics;
using ImGuiNET;

namespace Zoidberg.ui;

public enum LogType {
    Info,
    Warning,
    Error
}

internal struct Log {
    public string Message;
    public LogType Type;
}

public static class CustomConsole {
    private static readonly List<Log> Logs = [];
    private static Log[] _logsCopy = [];
    private static bool _shouldUpdateLogs;

    public static void Log(string message, LogType type = LogType.Info) {
        Logs.Add(new Log { Message = message, Type = type });
        
        if (Logs.Count > 1000)
            Logs.RemoveAt(0);
        
        _shouldUpdateLogs = true;
    }

    private static void UpdateLogs() {
        if (!_shouldUpdateLogs) return;
        
        _logsCopy = Logs.ToArray();
        _shouldUpdateLogs = false;
    }
    
    public static void Render() {
        var style = ImGui.GetStyle();

        ImGui.SeparatorText("Log");
        ImGui.PushStyleColor(ImGuiCol.ChildBg, style.Colors[(int)ImGuiCol.WindowBg]);
        ImGui.BeginChild("Logs", new Vector2(ImGui.GetColumnWidth(), -ImGui.GetFrameHeight()),
            ImGuiChildFlags.FrameStyle);

        foreach (var log in _logsCopy) {
            var color = log.Type switch {
                LogType.Info => style.Colors[(int)ImGuiCol.Text],
                LogType.Warning => new Vector4(1f, 1f, 0f, 1f),
                LogType.Error => new Vector4(1f, 0f, 0f, 1f),
                _ => style.Colors[(int)ImGuiCol.Text]
            };

            ImGui.PushStyleColor(ImGuiCol.Text, color);
            ImGui.TextWrapped($"> {log.Message}");
            ImGui.PopStyleColor();
        }

        UpdateLogs();
        
        if (ImGui.GetScrollY() >= ImGui.GetScrollMaxY()) 
            ImGui.SetScrollHereY(1.0f);
        
        ImGui.EndChild();
        ImGui.PopStyleColor();
    }
}