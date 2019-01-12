template <typename T>
struct Neighbours3x3 {
	T p11; T p12; T p13;
	T p21; T p22; T p23;
	T p31; T p32; T p33;

};

template <typename T>
__device__ T d_getPixel(const T* const src, int x, int y, int width, int height){
	int colaced_loc = (x + y*width);
	if (colaced_loc < 0) {
		return make_uchar1(0);
	}
	
	if (colaced_loc > width*height){
		return src[width*height];
	}

	return src[colaced_loc];
}

template <typename T>
__device__ void d_getNeighbours_8(
	const T* const src,
	int width, int height,
	int x, int y, 
	Neighbours3x3<T> &neighbours
){
	
	neighbours.p11 = d_getPixel(src, x-1, y-1, width, height);
	neighbours.p12 = d_getPixel(src, x, y-1, width, height);
	neighbours.p13 = d_getPixel(src, x+1, y-1, width, height);

	neighbours.p21 = d_getPixel(src, x-1, y, width, height);
	neighbours.p22 = d_getPixel(src, x, y, width, height);
	neighbours.p23 = d_getPixel(src, x+1, y, width, height);

	neighbours.p31 = d_getPixel(src, x-1, y+1, width, height);
	neighbours.p32 = d_getPixel(src, x, y+1, width, height);
	neighbours.p33 = d_getPixel(src, x+1, y+1, width, height);

}