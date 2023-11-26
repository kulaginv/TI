#include "opencv2/opencv.hpp"
#include <iostream>
using namespace cv;
int main(int, char**)
{	Mat img, GSimg, dst, abs_dst;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;
	VideoCapture vid(0);
		if(!vid.isOpened())
			{std::cout<<"La camera ne fonctionne pas\n";
	return -1;}
		namedWindow("Camera",WINDOW_AUTOSIZE);
		//vid.set(CAP_PROP_FRAME_WIDTH,320);
		//vid.set(CAP_PROP_FRAME_HEIGHT,180);
		vid.set(CAP_PROP_FPS,60);
		
		//vid.set(CAP_PROP_AUTO_EXPOSURE,0.25);
		//vid.set(CAP_PROP_EXPOSURE,0.001);
		//vid.set(CAP_PROP_CONTRAST,100);
		//vid.set(CAP_PROP_BRIGHTNESS,50);
		
		std::cout<<"La largeur est = "<<vid.get(CAP_PROP_FRAME_WIDTH)<<"\n"<<
					"La hauteur est = "<<vid.get(CAP_PROP_FRAME_HEIGHT)<<"\n"<<
					"Le nombre d’images par seconde est = "<<vid.get(CAP_PROP_FPS)<<"\n"<<
					"Le paramètre d’exposition est = "<<vid.get(CAP_PROP_EXPOSURE)<<"\n"<<
					"Le paramètre d’exposition auto est = "<<vid.get(CAP_PROP_AUTO_EXPOSURE)<<"\n"<<
					"Le contraste est = "<<vid.get(CAP_PROP_CONTRAST)<<"\n"<<
					"La luminosité est = "<<vid.get(CAP_PROP_BRIGHTNESS)<<"\n";
		while(1)
		{
			vid >> img;
			cvtColor(img, GSimg, COLOR_BGR2GRAY);
			Laplacian( GSimg, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
			convertScaleAbs( dst, abs_dst );
			imshow("Camera", abs_dst);
			if (waitKey(1)==27) break;
		}
		return 0;
}

