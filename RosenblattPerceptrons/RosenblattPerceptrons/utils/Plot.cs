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
    private static Vector2 _lastHighlightPos = Vector2.Zero;

    public static void Begin(string id, Vector2 size, float min, float max, string? label = null) {
        _cursor = ImGui.GetCursorScreenPos();
        _drawList = ImGui.GetWindowDrawList();
        _style = ImGui.GetStyle();
        _size = size;
        _min = min;
        _max = max;
        _label = label;

        ImGui.InvisibleButton(id, size);

        _drawList.AddRectFilled(_cursor, _cursor + size,
            ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.FrameBg]), _style.FrameRounding, 0);
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
        var innerSize = _size - _style.FramePadding * 2;
        var innerCursor = _cursor + _style.FramePadding;
        
        for (var i = 0; i < lines.Length - 1; i++) {
            var p1 = new Vector2(
                innerCursor.X + i * (innerSize.X / (lines.Length - 1)),
                innerCursor.Y + innerSize.Y - innerSize.Y * (lines[i] - _min) / (_max - _min));
            
            var p2 = new Vector2(
                innerCursor.X + (i + 1) * (innerSize.X / (lines.Length - 1)),
                innerCursor.Y + innerSize.Y - innerSize.Y * (lines[i + 1] - _min) / (_max - _min));
            
            if (fillColor.HasValue) {
                var p3 = p2 with { Y = innerCursor.Y + innerSize.Y };
                var p4 = p1 with { Y = innerCursor.Y + innerSize.Y };

                _drawList.AddQuadFilled(p1, p2, p3, p4, fillColor.Value);
            }

            _drawList.AddLine(p1, p2, color, 1);
        }

        if (ImGui.IsItemHovered()) OnHover(ref lines);
    }

    private static void OnHover(ref float[] lines) {
        var mousePos = ImGui.GetMousePos();
        var innerCursor = _cursor + _style.FramePadding;
        var innerSize = _size - _style.FramePadding * 2;
        
        var index = (int)((mousePos.X - innerCursor.X) / (innerSize.X / lines.Length));
        
        if (index < 0 || index >= lines.Length) return;

        var pos = new Vector2(innerCursor.X + index * (innerSize.X / (lines.Length - 1)),
            innerCursor.Y + innerSize.Y - innerSize.Y * (lines[index] - _min) / (_max - _min));

        if (_lastHighlightPos != Vector2.Zero) pos = _lastHighlightPos + (pos - _lastHighlightPos) * 0.02f;
        
        var value = lines[index];
        var text = $"Value: {value:F4}";
        var textSize = ImGui.CalcTextSize(text);
        var rectPos = pos + new Vector2(10, -10);
        var rectSize = textSize + new Vector2(10, 10);

        _drawList.AddCircleFilled(pos, 4, ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.Text]));
        _drawList.AddRectFilled(rectPos, rectPos + rectSize,
            ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.PopupBg]), _style.FrameRounding, 0);
        
        _drawList.AddText(rectPos + new Vector2(rectSize.X / 2 - textSize.X / 2, rectSize.Y / 2 - textSize.Y / 2),
            ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.Text]), text);
        
        _lastHighlightPos = pos;
    }

    public static void End() {
        if (_label is not null) {
            var textSize = ImGui.CalcTextSize(_label);

            _drawList.AddText(_cursor + new Vector2(_size.X / 2 - textSize.X / 2, 5),
                ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.Text]), _label);
        }
    }
}