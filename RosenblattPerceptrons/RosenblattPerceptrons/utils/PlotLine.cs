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
    private static Vector4 _padding = new(10, 10, 10, 32);

    public static void Begin(string id, Vector2 size, float min, float max, string? label = null) {
        _cursor = ImGui.GetCursorScreenPos();
        _drawList = ImGui.GetWindowDrawList();
        _style = ImGui.GetStyle();
        _size = size;
        _label = label;
        _padding = new Vector4(32, 10, 10, 10);
        _min = min;
        
        if (max <= min) _max = min + 1;
        else _max = max;

        ImGui.InvisibleButton(id, size);

        _drawList.AddRectFilled(_cursor, _cursor + size,
            ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.FrameBg]), _style.FrameRounding, 0);
    }

    public static void Annotations(Vector2 amount, bool showGrid = true) {
        var innerSize = _size - _style.FramePadding * 2 - new Vector2(_padding.X + _padding.Z, _padding.Y + _padding.W);
        var innerCursor = _cursor + _style.FramePadding + new Vector2(_padding.X, _padding.Y);
        
        if (showGrid) {
            for (var i = 1; i < amount.X; i++) {
                var x = innerCursor.X + i * (innerSize.X / amount.X);
                var y = innerCursor.Y;

                _drawList.AddLine(new Vector2(x, y), new Vector2(x, y + innerSize.Y),
                    ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.Border]), 1);
            }

            for (var i = 1; i < amount.Y; i++) {
                var x = innerCursor.X;
                var y = innerCursor.Y + i * (innerSize.Y / amount.Y);

                _drawList.AddLine(new Vector2(x, y), new Vector2(x + innerSize.X, y),
                    ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.Border]), 1);
            }
        }
        
        for (var i = 0; i <= amount.X; i++) {
            var x = innerCursor.X + i * (innerSize.X / amount.X);
            var y = innerCursor.Y + innerSize.Y;

            if (!showGrid) _drawList.AddLine(new Vector2(x, y), new Vector2(x, y - 5),
                ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.Border]), 1);

            var text = $"{i:F2}";
            var textSize = ImGui.CalcTextSize(text);

            _drawList.AddText(new Vector2(x - textSize.X / 2, y),
                ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.Border]), text);
        }

        for (var i = 0; i <= amount.Y; i++) {
            var x = innerCursor.X;
            var y = innerCursor.Y + innerSize.Y - i * (innerSize.Y / amount.Y);

            if (!showGrid) _drawList.AddLine(new Vector2(x, y), new Vector2(x + 5, y),
                ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.Border]), 1);
            
            float value = _min + i * ((_max - _min) / amount.Y);
            var text = value >= 1f ?  $"{value:F1}" : $"{value:F2}";
            var textSize = ImGui.CalcTextSize(text);
            
            _drawList.AddText(new Vector2(x - textSize.X - 5, y - textSize.Y / 2),
                ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.Border]), text);
        }
    }

    public static void Line(ref float[] lines, uint color, uint? fillColor = null) {
        var innerSize = _size - _style.FramePadding * 2 - new Vector2(_padding.X + _padding.Z, _padding.Y + _padding.W);
        var innerCursor = _cursor + _style.FramePadding + new Vector2(_padding.X, _padding.Y);

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

        if (ImGui.IsItemHovered()) OnLineHover(ref lines);
    }
    
    public static void Bar(ref float[] bars, uint color, uint fillColor) {
        var innerSize = _size - _style.FramePadding * 2 - new Vector2(_padding.X + _padding.Z, _padding.Y + _padding.W);
        var innerCursor = _cursor + _style.FramePadding + new Vector2(_padding.X, _padding.Y);

        for (var i = 0; i < bars.Length; i++) {
            var p1 = new Vector2(
                innerCursor.X + i * (innerSize.X / bars.Length),
                innerCursor.Y + innerSize.Y);

            var p2 = new Vector2(
                innerCursor.X + (i + 1) * (innerSize.X / bars.Length),
                innerCursor.Y + innerSize.Y - innerSize.Y * (bars[i] - _min) / (_max - _min));

            _drawList.AddRectFilled(p1, p2, fillColor);
            _drawList.AddRect(p1, p2, color, 0, 0);
        }

        if (ImGui.IsItemHovered()) OnBarHover(ref bars);
    }
    
    private static void OnBarHover(ref float[] bars) {
        var mousePos = ImGui.GetMousePos();
        var innerSize = _size - _style.FramePadding * 2 - new Vector2(_padding.X + _padding.Z, _padding.Y + _padding.W);
        var innerCursor = _cursor + _style.FramePadding + new Vector2(_padding.X, _padding.Y);

        var index = (int)((mousePos.X - innerCursor.X) / (innerSize.X / bars.Length));

        if (index < 0 || index >= bars.Length) {
            _lastHighlightPos = Vector2.Zero;
            return;
        }

        var barWidth = innerSize.X / bars.Length;
        var pos = new Vector2(innerCursor.X + index * barWidth + barWidth / 2,
            innerCursor.Y + innerSize.Y - innerSize.Y * (bars[index] - _min) / (_max - _min));

        if (_lastHighlightPos != Vector2.Zero) pos = _lastHighlightPos + (pos - _lastHighlightPos) * 0.02f;

        var value = bars[index];
        var text = $"Value: {value:F4}";
        var textSize = ImGui.CalcTextSize(text);
        var rectSize = textSize + new Vector2(10, 10);
        var rectPos = pos + new Vector2(10, -10);
        
        if (rectPos.X + rectSize.X > innerCursor.X + innerSize.X) rectPos.X = pos.X - rectSize.X - 10;
        if (rectPos.Y < innerCursor.Y) rectPos.Y = pos.Y + 10;

        _drawList.AddCircleFilled(pos, 4, ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.Text]));
        _drawList.AddRectFilled(rectPos, rectPos + rectSize,
            ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.PopupBg]), _style.FrameRounding, 0);

        _drawList.AddText(rectPos + new Vector2(rectSize.X / 2 - textSize.X / 2, rectSize.Y / 2 - textSize.Y / 2),
            ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.Text]), text);

        _lastHighlightPos = pos;
    }
    

    private static void OnLineHover(ref float[] lines) {
        var mousePos = ImGui.GetMousePos();
        var innerSize = _size - _style.FramePadding * 2 - new Vector2(_padding.X + _padding.Z, _padding.Y + _padding.W);
        var innerCursor = _cursor + _style.FramePadding + new Vector2(_padding.X, _padding.Y);

        var index = (int)((mousePos.X - innerCursor.X) / (innerSize.X / lines.Length));

        if (index < 0 || index >= lines.Length) {
            _lastHighlightPos = Vector2.Zero;
            return;
        }

        var pos = new Vector2(innerCursor.X + index * (innerSize.X / (lines.Length - 1)),
            innerCursor.Y + innerSize.Y - innerSize.Y * (lines[index] - _min) / (_max - _min));

        if (_lastHighlightPos != Vector2.Zero) pos = _lastHighlightPos + (pos - _lastHighlightPos) * 0.02f;

        var value = lines[index];
        var text = $"Value: {value:F4}";
        var textSize = ImGui.CalcTextSize(text);
        var rectSize = textSize + new Vector2(10, 10);
        var rectPos = pos + new Vector2(10, -10);
        
        if (rectPos.X + rectSize.X > innerCursor.X + innerSize.X) rectPos.X = pos.X - rectSize.X - 10;
        if (rectPos.Y < innerCursor.Y) rectPos.Y = pos.Y + 10;

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
            var labelRectPos = _cursor with { X = _cursor.X + _size.X / 2 - textSize.X / 2 };
            var labelRectSize = textSize + new Vector2(10, 10);
            var labelTextPos = labelRectPos + new Vector2(5, 5);

            _drawList.AddRectFilled(labelRectPos, labelRectPos + labelRectSize,
                ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.FrameBg]), _style.FrameRounding, 0);

            _drawList.AddText(labelTextPos, ImGui.ColorConvertFloat4ToU32(_style.Colors[(int)ImGuiCol.Text]), _label);
        }
    }
}