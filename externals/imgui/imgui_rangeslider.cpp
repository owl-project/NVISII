#include "imgui_rangeslider.h"
#include "imgui_internal.h"
// https://github.com/ocornut/imgui/issues/76
// Taken from: https://github.com/wasikuss/imgui/commit/a50515ace6d9a62ebcd69817f1da927d31c39bb1

namespace ImGui
{
    // Create text input in place of a slider (when CTRL+Clicking on slider)
    bool InputScalarAsWidgetReplacement(const ImRect& aabb, const char* label, ImGuiDataType data_type, void* data_ptr, ImGuiID id, int decimal_precision)
    {
        ImGuiContext& g = *GImGui;
        ImGuiWindow* window = GetCurrentWindow();

        // Our replacement widget will override the focus ID (registered previously to allow for a TAB focus to happen)
        SetActiveID(g.ActiveId, window);
        SetHoveredID(0);
        FocusableItemUnregister(window);

        char buf[32];
        // DataTypeFormatString(data_type, data_ptr, decimal_precision, buf, IM_ARRAYSIZE(buf));
        DataTypeFormatString(buf, IM_ARRAYSIZE(buf), data_type, data_ptr, ".4f");
        // char* buf, int buf_size, ImGuiDataType data_type, const void* p_data, const char* format
        bool text_value_changed = InputTextEx(label, "", buf, IM_ARRAYSIZE(buf), aabb.GetSize(), ImGuiInputTextFlags_CharsDecimal | ImGuiInputTextFlags_AutoSelectAll);
        if (g.ActiveId == 0)
        {
            // First frame
            IM_ASSERT(g.ActiveId == id);    // InputText ID expected to match the Slider ID (else we'd need to store them both, which is also possible)
            g.ActiveId = g.ActiveId;
            SetHoveredID(id);
        }
        else if (g.ActiveId != g.ActiveId)
        {
            // Release
            g.ActiveId = 0;
        }
        if (text_value_changed)
            return DataTypeApplyOpFromText(buf, GImGui->InputTextState.TextA.begin(), data_type, data_ptr, NULL);
        return false;
    }

    float RoundScalar( float value, const char *fmt )
    {
        char buf[64];
        sprintf( buf, fmt, value );
        return (float)atof( buf );
    }

    // Parse display precision back from the display format string
    int ParseFormatPrecision(const char* fmt, int default_precision)
    {
        int precision = default_precision;
        while ((fmt = strchr(fmt, '%')) != NULL)
        {
            fmt++;
            if (fmt[0] == '%') { fmt++; continue; } // Ignore "%%"
            while (*fmt >= '0' && *fmt <= '9')
                fmt++;
            if (*fmt == '.')
            {
                precision = atoi(fmt + 1);
                if (precision < 0 || precision > 10)
                    precision = default_precision;
                if (fmt[2] == 'e' || fmt[2] == 'E')
                    precision = 38;
            }
            else if (fmt[0] == 'e' || fmt[0] == 'E')
                precision = 38;
            break;
        }
        return precision;
    }

	// ~80% common code with ImGui::SliderBehavior
	bool RangeSliderBehavior(const ImRect& frame_bb, ImGuiID id, float* v1, float* v2, float v_min, float v_max, float power, int decimal_precision, ImGuiSliderFlags flags)
	{
		ImGuiContext& g = *GImGui;
		ImGuiWindow* window = GetCurrentWindow();
		const ImGuiStyle& style = g.Style;

		// Draw frame
		RenderFrame(frame_bb.Min, frame_bb.Max, GetColorU32(ImGuiCol_FrameBg), true, style.FrameRounding);

		const bool is_non_linear = (power < 1.0f - 0.00001f) || (power > 1.0f + 0.00001f);
		const bool is_horizontal = (flags & ImGuiSliderFlags_Vertical) == 0;

		const float grab_padding = 2.0f;
		const float slider_sz = is_horizontal ? (frame_bb.GetWidth() - grab_padding * 2.0f) : (frame_bb.GetHeight() - grab_padding * 2.0f);
		float grab_sz;
		if (decimal_precision > 0)
			grab_sz = ImMin(style.GrabMinSize, slider_sz);
		else
			grab_sz = ImMin(ImMax(1.0f * (slider_sz / ((v_min < v_max ? v_max - v_min : v_min - v_max) + 1.0f)), style.GrabMinSize), slider_sz);  // Integer sliders, if possible have the grab size represent 1 unit
		const float slider_usable_sz = slider_sz - grab_sz;
		const float slider_usable_pos_min = (is_horizontal ? frame_bb.Min.x : frame_bb.Min.y) + grab_padding + grab_sz*0.5f;
		const float slider_usable_pos_max = (is_horizontal ? frame_bb.Max.x : frame_bb.Max.y) - grab_padding - grab_sz*0.5f;

		// For logarithmic sliders that cross over sign boundary we want the exponential increase to be symmetric around 0.0f
		float linear_zero_pos = 0.0f;   // 0.0->1.0f
		if (v_min * v_max < 0.0f)
		{
			// Different sign
			const float linear_dist_min_to_0 = powf(fabsf(0.0f - v_min), 1.0f / power);
			const float linear_dist_max_to_0 = powf(fabsf(v_max - 0.0f), 1.0f / power);
			linear_zero_pos = linear_dist_min_to_0 / (linear_dist_min_to_0 + linear_dist_max_to_0);
		}
		else
		{
			// Same sign
			linear_zero_pos = v_min < 0.0f ? 1.0f : 0.0f;
		}

		// Process clicking on the slider
		bool value_changed = false;
		if (g.ActiveId == id)
		{
			static int lastDrag = -1;
			if (g.IO.MouseDown[0])
			{
				const float mouse_abs_pos = is_horizontal ? g.IO.MousePos.x : g.IO.MousePos.y;
				float clicked_t = (slider_usable_sz > 0.0f) ? ImClamp((mouse_abs_pos - slider_usable_pos_min) / slider_usable_sz, 0.0f, 1.0f) : 0.0f;
				if (!is_horizontal)
					clicked_t = 1.0f - clicked_t;

				float new_value;
				if (is_non_linear)
				{
					// Account for logarithmic scale on both sides of the zero
					if (clicked_t < linear_zero_pos)
					{
						// Negative: rescale to the negative range before powering
						float a = 1.0f - (clicked_t / linear_zero_pos);
						a = powf(a, power);
						new_value = ImLerp(ImMin(v_max, 0.0f), v_min, a);
					}
					else
					{
						// Positive: rescale to the positive range before powering
						float a;
						if (fabsf(linear_zero_pos - 1.0f) > 1.e-6f)
							a = (clicked_t - linear_zero_pos) / (1.0f - linear_zero_pos);
						else
							a = clicked_t;
						a = powf(a, power);
						new_value = ImLerp(ImMax(v_min, 0.0f), v_max, a);
					}
				}
				else
				{
					// Linear slider
					new_value = ImLerp(v_min, v_max, clicked_t);
				}

				// Round past decimal precision
				new_value = RoundScalar(new_value, /*decimal_precision*/ ".5f");
				if (*v1 != new_value || *v2 != new_value)
				{

					if (lastDrag == 1 || lastDrag == 2)
					{
						if (*v2 == new_value)
						{
							*v1 = new_value;
							//BOOST_LOG_TRIVIAL(info) << "lastdrag 2";
							lastDrag = 2; //left to right, just hit same spot
						}
					}
					else if(lastDrag == 0 || lastDrag == 3)
					{
						if (*v1 == new_value)
						{
							*v2 = new_value;
							//BOOST_LOG_TRIVIAL(info) << "lastdrag 3";
							lastDrag = 3; //right to left, just hit left spot
						}
					}

					if (fabsf(*v1 - new_value) == fabsf(*v2 - new_value))
					{
						if (*v1 == *v2)
						{
							if (*v1 != new_value)
							{
								if (new_value > *v1)
								{
									*v2 = new_value;
								}
								else
								{
									*v1 = new_value;
								}
							}
						} 
						//else

						if (lastDrag == 1)
						{
							*v1 = new_value;
						}
						else if (lastDrag == 0)
						{
							*v2 = new_value;
						}
						
					}
					else if (fabsf(*v1 - new_value) < fabsf(*v2 - new_value))
					{
						*v1 = new_value;
						lastDrag = 1;
					}
					else
					{
						*v2 = new_value;
						lastDrag = 0;
					}
					value_changed = true;
				}
			}
			else
			{
				ClearActiveID();
				lastDrag = -1;
			}
		}

		// Calculate slider grab positioning
		// float grab_t = SliderBehaviorCalcRatioFromValue(*v1, v_min, v_max, power, linear_zero_pos);
        // ImS32, ImS32, float
        float grab_t = SliderCalcRatioFromValueT<ImS32,float>(ImGuiDataType_Float, *v1, v_min, v_max, power, linear_zero_pos);

		// float grab_t = SliderCalcRatioFromValueT<float*, float>(ImGuiDataType_Float, *v1, v_min, v_max, power, linear_zero_pos);

        // (ImGuiDataType data_type, T v, T v_min, T v_max, float power, float linear_zero_pos);
        
		// Draw
		if (!is_horizontal)
			grab_t = 1.0f - grab_t;
		float grab_pos = ImLerp(slider_usable_pos_min, slider_usable_pos_max, grab_t);
		ImRect grab_bb1;
		if (is_horizontal)
			grab_bb1 = ImRect(ImVec2(grab_pos - grab_sz*0.5f, frame_bb.Min.y + grab_padding), ImVec2(grab_pos + grab_sz*0.5f, frame_bb.Max.y - grab_padding));
		else
			grab_bb1 = ImRect(ImVec2(frame_bb.Min.x + grab_padding, grab_pos - grab_sz*0.5f), ImVec2(frame_bb.Max.x - grab_padding, grab_pos + grab_sz*0.5f));
		window->DrawList->AddRectFilled(grab_bb1.Min, grab_bb1.Max, GetColorU32(g.ActiveId == id ? ImGuiCol_SliderGrabActive : ImGuiCol_SliderGrab), style.GrabRounding);

		// Calculate slider grab positioning
		grab_t = SliderCalcRatioFromValueT<ImS32,float>(ImGuiDataType_Float,*v2, v_min, v_max, power, linear_zero_pos);

		// Draw
		if (!is_horizontal)
			grab_t = 1.0f - grab_t;
		grab_pos = ImLerp(slider_usable_pos_min, slider_usable_pos_max, grab_t);
		ImRect grab_bb2;
		if (is_horizontal)
			grab_bb2 = ImRect(ImVec2(grab_pos - grab_sz*0.5f, frame_bb.Min.y + grab_padding), ImVec2(grab_pos + grab_sz*0.5f, frame_bb.Max.y - grab_padding));
		else
			grab_bb2 = ImRect(ImVec2(frame_bb.Min.x + grab_padding, grab_pos - grab_sz*0.5f), ImVec2(frame_bb.Max.x - grab_padding, grab_pos + grab_sz*0.5f));
		window->DrawList->AddRectFilled(grab_bb2.Min, grab_bb2.Max, GetColorU32(g.ActiveId == id ? ImGuiCol_SliderGrabActive : ImGuiCol_SliderGrab), style.GrabRounding);

		ImRect connector(grab_bb1.Min, grab_bb2.Max);
		connector.Min.x += grab_sz;
		connector.Min.y += grab_sz*0.3f;
		connector.Max.x -= grab_sz;
		connector.Max.y -= grab_sz*0.3f;

		window->DrawList->AddRectFilled(connector.Min, connector.Max, GetColorU32(ImGuiCol_RangeSliderBar), style.GrabRounding);

		return value_changed;
	}

	// ~95% common code with ImGui::SliderFloat
	bool RangeSliderFloat(const char* label, float* v1, float* v2, float v_min, float v_max, const char* display_format, float power)
	{
		ImGuiWindow* window = GetCurrentWindow();
		if (window->SkipItems)
			return false;

		ImGuiContext& g = *GImGui;
		const ImGuiStyle& style = g.Style;
		const ImGuiID id = window->GetID(label);
		const float w = CalcItemWidth();

		const ImVec2 label_size = CalcTextSize(label, NULL, true);
		const ImRect frame_bb(window->DC.CursorPos, ImVec2(window->DC.CursorPos.x + w, window->DC.CursorPos.y + label_size.y + style.FramePadding.y*2.0f));
		const ImRect total_bb(frame_bb.Min, ImVec2(frame_bb.Max.x + label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f, frame_bb.Max.y + 0.0f));

		// NB- we don't call ItemSize() yet because we may turn into a text edit box below
		if (!ItemAdd(total_bb, id))
		{
			ItemSize(total_bb, style.FramePadding.y);
			return false;
		}

		const bool hovered = ItemHoverable(frame_bb, id);
		if (hovered)
			SetHoveredID(id);

		if (!display_format)
			display_format = "(%.3f, %.3f)";
		int decimal_precision = ParseFormatPrecision(display_format, 3);

		// Tabbing or CTRL-clicking on Slider turns it into an input box
		bool start_text_input = false;
		const bool tab_focus_requested = FocusableItemRegister(window, g.ActiveId == id);
		if (tab_focus_requested || (hovered && g.IO.MouseClicked[0]))
		{
			SetActiveID(id, window);
			FocusWindow(window);

			if (tab_focus_requested || g.IO.KeyCtrl)
			{
				start_text_input = true;
				g.TempInputTextId = 0;
			}
		}
		if (start_text_input || (g.ActiveId == id && g.TempInputTextId == id))
			return InputScalarAsWidgetReplacement(frame_bb, label, ImGuiDataType_Float, v1, id, decimal_precision);

		ItemSize(total_bb, style.FramePadding.y);

		// Actual slider behavior + render grab
		const bool value_changed = RangeSliderBehavior(frame_bb, id, v1, v2, v_min, v_max, power, decimal_precision, 0);

		// Display value using user-provided display format so user can add prefix/suffix/decorations to the value.
		char value_buf[64];
		const char* value_buf_end = value_buf + ImFormatString(value_buf, IM_ARRAYSIZE(value_buf), display_format, *v1, *v2);
		RenderTextClipped(frame_bb.Min, frame_bb.Max, value_buf, value_buf_end, NULL, ImVec2(0.5f, 0.5f));

		if (label_size.x > 0.0f)
			RenderText(ImVec2(frame_bb.Max.x + style.ItemInnerSpacing.x, frame_bb.Min.y + style.FramePadding.y), label);

		return value_changed;
	}

} // namespace ImGui