using System.Numerics;
using ImGuiNET;

namespace RosenblattPerceptrons.utils;

public static class Plot {
    private static Vector2 _cursor;
    private static ImDrawListPtr _drawList;
    private static ImGuiStylePtr _style;
    private static Vector2 _size;
    private static float _min;
    private static float _max;
    private static string? _label;

    public static void Begin(string id, Vector2 size, float min, float max, string? label = null) {
        _cursor = ImGui.GetCursorScreenPos();
        _drawList = ImGui.GetWindowDrawList();
        _style = ImGui.GetStyle();
        _size = size;
        _min = min;
        _max = max;
        _label = label;

        ImGui.InvisibleButton(id, size);
        
        if (ImGui.IsItemHovered()) {
            OnHover();
        };

        _drawList.AddRectFilled(_cursor, _cursor + size,
            ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.FrameBg]), _style.FrameRounding, 0);
    }

    private static void OnHover() {
        var io = ImGui.GetIO();
        var mousePos = io.MousePos;
        
        var current = new Vector2(
            (mousePos.X - _cursor.X) / _size.X * (_max - _min) + _min,
            _max - (mousePos.Y - _cursor.Y) / _size.Y * (_max - _min));
        
        var text = $"({current.X:0.00}, {current.Y:0.00})";
        
        var textSize = ImGui.CalcTextSize(text);
        var padding = 5;
        
        var p1 = mousePos + new Vector2(padding, padding);
        var p2 = p1 + textSize + new Vector2(padding, padding);
        
        _drawList.AddRectFilled(p1, p2,
            ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.FrameBg]), _style.FrameRounding, 0);
        
        _drawList.AddText(p1, ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.Text]), text);
    }

    public static void Annotations(Vector2 amount) {
        for (var i = 1; i <= amount.X; i++) {
            var x = _cursor.X + i * (_size.X / amount.X) - 1;
            var y = _cursor.Y + _size.Y;

            _drawList.AddLine(new Vector2(x, y), new Vector2(x, y - 5),
                ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.Border]), 1);
        }

        for (var i = 1; i <= amount.Y; i++) {
            var x = _cursor.X;
            var y = _cursor.Y + _size.Y - i * (_size.Y / amount.Y) - 1;

            _drawList.AddLine(new Vector2(x, y), new Vector2(x + 5, y),
                ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.Border]), 1);
        }
    }

    public static void Line(ref float[] lines, uint color, uint? fillColor = null) {
        for (var i = 0; i < lines.Length - 1; i++) {
            var p1 = new Vector2(
                _cursor.X + i * (_size.X / (lines.Length - 1)),
                _cursor.Y + _size.Y - _size.Y * (lines[i] - _min) / (_max - _min));

            var p2 = new Vector2(
                _cursor.X + (i + 1) * (_size.X / (lines.Length - 1)),
                _cursor.Y + _size.Y - _size.Y * (lines[i + 1] - _min) / (_max - _min));

            if (fillColor.HasValue) {
                var p3 = p2 with { Y = _cursor.Y + _size.Y };
                var p4 = p1 with { Y = _cursor.Y + _size.Y };

                _drawList.AddQuadFilled(p1, p2, p3, p4, fillColor.Value);
            }

            _drawList.AddLine(p1, p2, color, 1);
        }
    }

    public static void End() {
        if (_label is not null) {
            var textSize = ImGui.CalcTextSize(_label);

            _drawList.AddText(_cursor + new Vector2(_size.X / 2 - textSize.X / 2, 5),
                ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.Text]), _label);
        }
    }
}