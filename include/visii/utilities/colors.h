#pragma once
#include <array>
namespace Colors {
	inline std::array<float, 3> hsvToRgb(std::array<float, 3> hsv) {
		std::array<float, 3> rgb = {};
		int i = (int)floorf(hsv[0] * 6.0f);
		float f = hsv[0] * 6 - i;
		float p = hsv[2] * (1 - hsv[1]);
		float q = hsv[2] * (1 - f * hsv[1]);
		float t = hsv[2] * (1 - (1 - f) * hsv[1]);

		switch (i % 6) {
		case 0: rgb[0] = hsv[2], rgb[1] = t, rgb[2] = p; break;
		case 1: rgb[0] = q, rgb[1] = hsv[2], rgb[2] = p; break;
		case 2: rgb[0] = p, rgb[1] = hsv[2], rgb[2] = t; break;
		case 3: rgb[0] = p, rgb[1] = q, rgb[2] = hsv[2]; break;
		case 4: rgb[0] = t, rgb[1] = p, rgb[2] = hsv[2]; break;
		case 5: rgb[0] = hsv[2], rgb[1] = p, rgb[2] = q; break;
		}

		return rgb;
	}

	// static inline glm::vec4 black = glm::vec4(0.0, 0.0, 0.0, 1.0);
	// static inline glm::vec4 white = glm::vec4(1.0, 1.0, 1.0, 1.0);
	// static inline glm::vec4 darkGrey = glm::vec4(.05, .05, .05, 1.0);
	// static inline glm::vec4 red = glm::vec4(1.0, 0.0, 0.0, 1.0);
	// static inline glm::vec4 green = glm::vec4(0.0, 1.0, 0.0, 1.0);
	// static inline glm::vec4 blue = glm::vec4(0.0, 0.0, 1.0, 1.0);
}