
namespace cucv{

	enum ChannelCodes {
		GRAY = 0,
		RGB = 2,
		RGBA = 1
	}; // enum ChannelCodes

	enum ChannelConversionCodes {
		RGB2GRAY = 0,
		RGBA2GRAY = 1,
		GRAY2RGB = 2,
		GRAY2RGBA = 3
	};

	enum EdgeCodes {
		SIMPLE = 0,
		SOBEL = 1,
		CANNY = 2
	};

} // namespace cucv