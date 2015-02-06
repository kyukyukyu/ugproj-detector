#include <iostream>
#include "optical_flow.h"
#include "gaussian_pyramid.h"


namespace opticalflow {
MCImageDoubleX OpticalFlow::LapPara;

void OpticalFlow::SmoothFlowSOR(const MCImageDoubleX &Im1, const MCImageDoubleX &Im2, MCImageDoubleX& warpIm2, MCImageDoubleX& u, MCImageDoubleX& v,
    double alpha, int nOuterFPIterations, int nInnerFPIterations, int nSORIterations) {
  MCImageDoubleX mask;
  MCImageDoubleX imdx,imdy, imdt;
  int width,height,nChannels,nPixels;
  width=Im1.width();
  height=Im1.height();
  nChannels=Im1.num_channels();
  nPixels=Im1.size();

  MCImageDoubleX du(width,height), dv(width,height), 
                 uu(width,height), vv(width,height), 
                 ux(width,height), uy(width,height), 
                 vx(width,height), vy(width,height);

  uu.fill(0), vv.fill(0);
  ux.fill(0), uy.fill(0), vx.fill(0), vy.fill(0);

  MCImageDoubleX Phi_1st(width,height), Psi_1st(width,height,nChannels);

  MCImageDoubleX imdxy(width,height), 
                 imdx2(width,height), 
                 imdy2(width,height), 
                 imdtdx(width,height), 
                 imdtdy(width,height);
  MCImageDoubleX ImDxy, ImDx2, ImDy2, ImDtDx, ImDtDy;
  MCImageDoubleX foo1, foo2;

//  double prob1,prob2,prob11,prob22;

  double varepsilon_phi=pow(0.001,2);
  double varepsilon_psi=pow(0.001,2);

  for (int count = 0; count < nOuterFPIterations; count++) {
    getDxs(imdx, imdy, imdt, Im1, warpIm2);
    
    genInImageMask(mask,u,v);

    du.fill(0);
    dv.fill(0);

    for (int hh = 0; hh < nInnerFPIterations; hh++) {
      if (hh == 0) {
        uu = u;
        vv = v;
      }
      else {
        add(u,du,uu);
        add(v,dv,vv);
      }
      dx(uu, ux);
      dy(uu, uy);
      dx(vv, vx);
      dy(vv, vy);

      double temp;
      double power_alpha = 0.5;

      // Regularization term
      Phi_1st.fill(0);
      for (int r=0; r<height; r++) {
        for (int c=0; c<width; c++) {
          temp = ux(c,r,0)*ux(c,r,0) + uy(c,r,0)*uy(c,r,0) + 
            vx(c,r,0)*vx(c,r,0) + vy(c,r,0)*vy(c,r,0);
          Phi_1st(c,r,0) = 0.5 / sqrt(temp+varepsilon_phi);
        }
      }

      // Data term
      Psi_1st.fill(0);
      double _a = 10000, _b = 0.1;
      if (nChannels == 1) {
        for (int r=0; r<height; r++) {
          for (int c=0; c<width; c++) {
            temp = imdt(c,r,0) + imdx(c,r,0)*du(c,r,0) + imdy(c,r,0)*dv(c,r,0);
            temp *= temp;
            if (LapPara(0,0,0) < 1E-20)
              continue;
            Psi_1st(c,r,0) = 0.5 / sqrt(temp+varepsilon_psi);
          }
        }
      }
      else {
        for (int r=0; r<height; r++) {
          for (int c=0; c<width; c++) {
            for (int k=0; k<nChannels; k++) {
              temp = imdt(c,r,k) + imdx(c,r,k)*du(c,r,0) + imdy(c,r,k)*dv(c,r,0);
              temp *= temp;
            // we fix noise model to Laplacian
              if (LapPara(0,0,k) < 1E-20)
                continue;
              Psi_1st(c,r,k) = 0.5 / sqrt(temp+varepsilon_psi);
            }
          }
        }
      }

      //ImDxy = Psi_1st * imdx * imdy;
      ImDxy = Mult(Psi_1st,imdx,imdy);
			ImDx2 = Mult(Psi_1st,imdx,imdx);
			ImDy2 = Mult(Psi_1st,imdy,imdy);
			ImDtDx = Mult(Psi_1st,imdx,imdt);
			ImDtDy = Mult(Psi_1st,imdy,imdt);
			//ImDtDy.Multiply(Psi_1st,imdy,imdt);

			if(nChannels>1)
			{
        collapse(ImDxy, imdxy);
        collapse(ImDx2, imdx2);
        collapse(ImDy2, imdy2);
        collapse(ImDtDx, imdtdx);
        collapse(ImDtDy, imdtdy);
			}
			else
			{
				imdxy = ImDxy;
				imdx2 = ImDx2;
				imdy2 = ImDy2;
        imdtdx = ImDtDx;
				imdtdy = ImDtDy;
			}
			// laplacian filtering of the current flow field
		  Laplacian(foo1,u,Phi_1st);
			Laplacian(foo2,v,Phi_1st);

      for (int r=0; r<height; r++) {
        for (int c=0; c<width; c++) {
          imdtdx(c,r,0) = -imdtdx(c,r,0)-alpha*foo1(c,r,0);
          imdtdy(c,r,0) = -imdtdy(c,r,0)-alpha*foo2(c,r,0);
        }
      }

			// here we start SOR

			// set omega
			double omega = 1.8;

			du.fill(0);
			dv.fill(0);

			for(int k=0; k<nSORIterations; k++)
				for(int r=0; r<height; r++)
					for(int c=0; c<width; c++)
					{
						double sigma1 = 0, sigma2 = 0, coeff = 0;
                        double _weight;
						
						if(c>0)
						{
              _weight = Phi_1st(c-1,r,0);
              sigma1 += _weight * du(c-1,r,0);
              sigma2 += _weight * dv(c-1,r,0);
              coeff += _weight;
						}
						if(c<width-1)
						{
              _weight = Phi_1st(c,r,0);
              sigma1 += _weight * du(c+1,r,0);
              sigma2 += _weight * dv(c+1,r,0);
							coeff += _weight;
						}
						if(r>0)
						{
              _weight = Phi_1st(c,r-1,0);
              sigma1 += _weight * du(c,r-1,0);
              sigma2 += _weight * dv(c,r-1,0);
							coeff   += _weight;
						}
						if(r<height-1)
						{
              _weight = Phi_1st(c,r,0);
              sigma1 += _weight * du(c,r+1,0);
              sigma2 += _weight * dv(c,r+1,0);
							coeff   += _weight;
						}
						sigma1 *= -alpha;
						sigma2 *= -alpha;
						coeff *= alpha;
						 // compute du, dv
            sigma1 += imdxy(c,r,0) * dv(c,r,0);
            du(c,r,0) = (1-omega)*du(c,r,0) + omega/(imdx2(c,r,0)+alpha*0.05+coeff)*(imdtdx(c,r,0)-sigma1);
            sigma2 += imdxy(c,r,0) * du(c,r,0);
            dv(c,r,0) = (1-omega)*dv(c,r,0) + omega/(imdy2(c,r,0)+alpha*0.05+coeff)*(imdtdy(c,r,0)-sigma2);
					}
		}
		add(u,du,u);
		add(v,dv,v);

    warpFL(warpIm2,Im1,Im2,u,v);
    estLaplacianNoise(Im1,warpIm2,LapPara);
  }

}

void OpticalFlow::Coarse2FineFlow(MCImageDoubleX& vx, MCImageDoubleX& vy, MCImageDoubleX& warpI2, const MCImageDoubleX &Im1, const MCImageDoubleX &Im2, 
      double alpha, double ratio, int minWidth, int nOuterFPIterations, int nInnerFPIterations, int nCGIterations) {

  GaussianPyramid img_pyramid1;
  GaussianPyramid img_pyramid2;

  LapPara.Resize(1, 1, Im1.num_channels()+2);
  for (int i=0; i<LapPara.num_channels(); i++)
    LapPara(0, 0, i) = 0.02;

  img_pyramid1.ConstructPyramid(Im1, ratio, minWidth);
  img_pyramid2.ConstructPyramid(Im2, ratio, minWidth);

  MCImageDoubleX Image1, Image2, WarpImage2;

  for (int k = img_pyramid1.nlevels()-1; k >= 0; k--) {
    int width = img_pyramid1.Image(k).width();;
    int height = img_pyramid1.Image(k).height();
    
    im2feature(Image1, img_pyramid1.Image(k));
    im2feature(Image2, img_pyramid2.Image(k));

    if (k==img_pyramid1.nlevels()-1) {
      vx.Resize(width, height, 1);
      vy.Resize(width, height, 1);
      vx.fill(0);
      vy.fill(0);
      WarpImage2.Copy(Image2);
    }
    else {
      MCImageDoubleX foo;
      ResizeImage(vx, foo, width, height);
      vx.Copy(foo);
      vx *= 1/ratio;
      ResizeImage(vy, foo, width, height);
      vy.Copy(foo);
      vy *= 1/ratio;
      //warping with bicubic interpolation
      warpFL(WarpImage2,Image1,Image2,vx,vy);

    }

    SmoothFlowSOR(Image1,Image2,WarpImage2,vx,vy,
        alpha,nOuterFPIterations+k,nInnerFPIterations,nCGIterations+k*3);
  }
  warpFL(warpI2,Im1,Im2,vx,vy);
}

void OpticalFlow::im2feature(MCImageDoubleX &imfeature, const MCImageDoubleX &im) {
  int width=im.width();
  int height=im.height();
  int nchannels=im.num_channels();

  if (nchannels == 1) {
    imfeature.Resize(width, height, 3);
    imfeature.fill(0);
    MCImageDoubleX imdx, imdy;
    dx(im, imdx, true);
    dy(im, imdy, true);

    for (int r=0; r<height; r++)
      for (int c=0; c<width; c++)
      {
        imfeature(r, c, 0) = im(r, c, 0);
        imfeature(r, c, 1) = imdx(r, c, 0);
        imfeature(r, c, 2) = imdy(r, c, 0);
      }
  }
  else if (nchannels == 3) {
    MCImageDoubleX grayImage;
    desaturate(im, grayImage, BGR);

    imfeature.Resize(width, height, 5);
    imfeature.fill(0);

    MCImageDoubleX imdx, imdy;
    dx(grayImage, imdx, true);
    dy(grayImage, imdy, true);

    for (int i=0; i<height; i++)
      for (int j=0; j<width; j++) {
        imfeature(j, i, 0) = grayImage(j, i, 0);
        imfeature(j, i, 1) = imdx(j, i, 0);
        imfeature(j, i, 2) = imdy(j, i, 0);
        imfeature(j, i, 3) = im(j, i, 1) - im(j, i, 0);
        imfeature(j, i, 4) = im(j, i, 1) - im(j, i, 2);
      }
  }
  else
    imfeature.Copy(im);
}

void OpticalFlow::getDxs(MCImageDoubleX& imdx, MCImageDoubleX& imdy, MCImageDoubleX& imdt, const MCImageDoubleX& im1, const MCImageDoubleX& im2) {
  Eigen::ArrayXXd gFilter;
  gFilter.resize(1, 5);
  gFilter << .02, .11, .74, .11, .02;
//  gFilter(0, 0) = 0.02;  gFilter(0, 1) = 0.11;  gFilter(0, 2) = 0.74;  gFilter(0, 3) = 0.11;  gFilter(0, 4) = 0.02;

  MCImageDoubleX Im1, Im2, Im;

  imfilter_hv(im1, Im1, gFilter, 2, gFilter, 2);
  imfilter_hv(im2, Im2, gFilter, 2, gFilter, 2);
  Im.Copy(Im1);
  multiplyWith(Im, 0.4);
  add(0.6, Im2, Im);

  dx(Im, imdx, true);
  dy(Im, imdy, true);
  subtract(Im2, Im1, imdt);
}


const MCImageDoubleX OpticalFlow::Mult(const MCImageDoubleX& im1, const MCImageDoubleX& im2, const MCImageDoubleX& im3) {
  // Assume that all arguments have the same dimension
  int height = im1.height();
  int width = im1.width();
  int channels = im1.num_channels();
  MCImageDoubleX result(width, height, channels);

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      for (int k = 0; k < channels; k++) {
        result(j, i, k) = im1(j, i, k) * im2(j, i, k) * im3(j, i, k);
      }
    }
  }
  return result;
}

void OpticalFlow::multiplyWith(MCImageDoubleX& image, double ratio) {
  int height = image.height();
  int width = image.width();
  int channels = image.num_channels();

  for (int i = 0; i < height; i++) {
  for (int j = 0; j < width; j++) {
    for (int k = 0; k < channels; k++) {
        image(j, i, k) = ratio * image(j, i, k);
    }
  }
  }
}

void OpticalFlow::add(MCImageDoubleX& img1, MCImageDoubleX& img2, MCImageDoubleX& result) {
  // Assume that all arguments have the same dimension
  int height = result.height();
  int width = result.width();
  int channels = result.num_channels();

  if (!matchDimension(img1, result))
    result.Resize(width, height, channels);

  for (int i = 0; i < height; i++){
  for (int j = 0; j < width; j++){
    for (int k = 0; k < channels; k++){
    result(j, i, k) = img1(j, i, k) + img2(j, i, k);
    }
  }
  }
}

void OpticalFlow::add(double val, MCImageDoubleX& img, MCImageDoubleX& result) {
  int height = img.height();
  int width = img.width();
  int channels = img.num_channels();

  if (!matchDimension(img, result)) {
    MCImageDoubleX foo;
    ResizeImage(result, foo, width, height);
    result.Copy(foo);
  }

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      for (int k = 0; k < channels; k++) {
        result(j, i, k) += img(j, i, k) * val;
      }
    }
  }
}

void OpticalFlow::subtract(MCImageDoubleX& img1, MCImageDoubleX& img2, MCImageDoubleX& result) {
  // Assume that all arguments have the same dimension
  int height = img1.height();
  int width = img1.width();
  int channels = img1.num_channels();

  if (!matchDimension(img1, result))
    result.Resize(width, height, channels);

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      for (int k = 0; k < channels; k++) {
        double tmp1 = img1(j, i, k), tmp2 = img2(j, i, k); 
        result(j, i, k) = img1(j, i, k) - img2(j, i, k);
      }
    }
  }
}

void OpticalFlow::collapse(const MCImageDoubleX& in, MCImageDoubleX& out) {
  int height = in.height();
  int width = in.width();

  if (!matchDimension(in, out))
    out.Resize(width, height, 1);

  for (int r=0; r<height; r++) {
    for (int c=0; c<width; c++) {
      out(c,r,0) = in(c,r).mean();
    }
  }
}

void OpticalFlow::hfiltering(const MCImageDoubleX& pSrcImage, MCImageDoubleX& pDstImage, int width, int height, int nChannels, const Eigen::ArrayXXd& pfilter1D, int fsize) {
  pDstImage.Resize(width, height, nChannels);
  pDstImage.fill(0);

  double w;
  int r, c, l, k, cc;
  
  for(r=0 ; r<height ; r++) {
    for(c=0 ; c<width ; c++) {
      for(l=-fsize ; l<=fsize ; l++) {
        w = pfilter1D(0, l+fsize);
        cc = EnforceRange(c+l, width);
        for(k=0 ; k<nChannels ; k++) {
          pDstImage(c, r, k) += pSrcImage(cc, r, k) * w;
        }
      }
    }
  }
}

void OpticalFlow::vfiltering(const MCImageDoubleX& pSrcImage, MCImageDoubleX& pDstImage, int width, int height, int nChannels, const Eigen::ArrayXXd& pfilter1D, int fsize) {
	pDstImage.Resize(width, height, nChannels);
  pDstImage.fill(0);

	double w;
	int r, c, l, k, rr;
	
	for(r=0 ; r<height ; r++) {
		for(c=0 ; c<width ; c++) {
			for(l=-fsize ; l<=fsize ; l++) {
				w = pfilter1D(0, l+fsize);
				rr = EnforceRange(r+l, height);
				for(k=0 ; k<nChannels ; k++) {
					pDstImage(c, r, k) += pSrcImage(c, rr, k) * w;
				}
			}
		}
	}
}

void OpticalFlow::imfilter_hv(const MCImageDoubleX& input, MCImageDoubleX& result, const Eigen::ArrayXXd& hfilter, int hfsize, const Eigen::ArrayXXd& vfilter, int vfsize) {
  //ConvolveImage(input, hfilter, &result);
  //ConvolveImage(input, vfilter.transpose(), &result);
  MCImageDoubleX tmp_image(input.width(), input.height(), input.num_channels());

  OpticalFlow::hfiltering(input, tmp_image, input.width(), input.height(), input.depth(), hfilter, hfsize);
  OpticalFlow::vfiltering(tmp_image, result, input.width(), input.height(), input.depth(), vfilter, vfsize);
}

void OpticalFlow::dx(const MCImageDoubleX& input, MCImageDoubleX& result, bool IsAdvancedFilter) {
  int height = input.height();
  int width = input.width();
  int channels = input.num_channels();

  int j, i, k, offset;

  if (!matchDimension(input, result))
    result.Resize(width, height, 1);

  if (!IsAdvancedFilter) {
    for (i = 0; i < height; i++) {
      for (j = 0; j < width - 1; j++) {
        offset = i*height + j;
        for (k = 0; k < channels; k++) {
          result(j, i, k) = input(j+1, i, k) - input(j, i, k);
       //   data(offset, k) = input(offset + 1, k) - input(offset, k);
        }
      }
    }
  }
  else {
    Eigen::ArrayXXd xFilter;
    xFilter.resize(1, 5);
    xFilter << 1, -8, 0, 8, -1;
    for (i = 0; i < 5; i++) {
      xFilter(0, i) = xFilter(0, i) / 12;
    }
  
    // filtering is needed
   // ConvolveImage(input, xFilter, &result);
    hfiltering(input, result, width, height, channels, xFilter, 2);
  }
}

void OpticalFlow::dy(const MCImageDoubleX& input, MCImageDoubleX& result, bool IsAdvancedFilter) {
  MCImageDoubleX& data = result;
  int height = input.height();
  int width = input.width();
  int channels = input.num_channels();

  int j, i, k, offset;

  if (!matchDimension(input, result))
    result.Resize(width, height, 1);

  if (IsAdvancedFilter == false){
    for (i = 0; i < height - 1; i++){
      for (j = 0; j < width; j++){
        offset = i*height + j;
        for (k = 0; k < channels; k++){
          data(j, i, k) = input(j, i+1, k) - input(j, i, k);
       //   data(offset, k) = input(offset + width, k) - input(offset, k);
        }
      }
    }
  }
  else{
    Eigen::ArrayXXd yFilter;
    yFilter.resize(1, 5);
    yFilter << 1, -8, 0, 8, -1;
  for (i = 0; i < 5; i++){
    yFilter(0, i) = yFilter(0, i) / 12;
  }
    // filtering is needed
  vfiltering(input, result, width, height, channels, yFilter, 2);
  //ConvolveImage(input, yFilter.transpose(), &result);
  }
}

void OpticalFlow::dxx(const MCImageDoubleX& input, MCImageDoubleX& result) {
  MCImageDoubleX pDstData = result;
  int height = input.height();
  int width = input.width();
  //int imgSize = height * width;
  int nChannels = input.num_channels();

/*  if (nChannels == 1) {
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      int offset = i * width + j;
      if (j == 0) {
        pDstData[offset] = input[offset] - input[offset + 1];
        pDstData(j,i) = input(j,i) - input(j,i+1);
        continue;
      }
      if (j == width - 1) {
        pDstData[offset] = input[offset] - input[offset - 1];
        continue;
      }
      pDstData[offset] = input[offset] * 2 - input[offset - 1] - input[offset + 1];
      }
    }
  }
  else */
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (j == 0) {
        for (int k = 0; k < nChannels; k++) {
          pDstData(j, i, k) = input(j, i, k) - input(j+1, i, k);
        }
        continue;
      }
      if (j == width - 1) {
        for (int k = 0; k < nChannels; k++) {
        pDstData(j, i, k) = input(j, i, k) - input(j-1, i, k);
        }
      }
      for (int k = 0; k < nChannels; k++) {
        pDstData(j, i, k) = input(j, i, k) * 2 - input(j+1, i, k) - input(j-1, i, k);
      }
    }
  }
}

void OpticalFlow::dyy(const MCImageDoubleX& input, MCImageDoubleX& result) {
  MCImageDoubleX pDstData = result;
  int height = input.height();
  int width = input.width();
  //int imgSize = height * width;
  int nChannels = input.num_channels();

/*  if (nChannels == 1){
  for (int i = 0; i < height; i++){
    for (int j = 0; j < width; j++){
    int offset = i * width + j;
    if (i == 0){
      pDstData[offset] = input[offset] - input[offset + width];
      continue;
    }
    if (i == height - 1){
      pDstData[offset] = input[offset] - input[offset - width];
      continue;
    }
    pDstData[offset] = input[offset] * 2 - input[offset - width] - input[offset + width];
    }
  }
  }
  else */
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (j == 0) {
        for (int k = 0; k < nChannels; k++) {
          pDstData(j, i, k) = input(j, i, k) - input(j, i+1, k);
        }
        continue;
      }
      if (j == width - 1) {
        for (int k = 0; k < nChannels; k++) {
        pDstData(j, i, k) = input(j, i, k) - input(j, i-1, k);
        }
      }
      for (int k = 0; k < nChannels; k++) {
        pDstData(j, i, k) = input(j, i, k) * 2 - input(j, i+1, k) - input(j, i-1, k);
      }
    }   
  }
}

bool OpticalFlow::matchDimension(const MCImageDoubleX& img1, const MCImageDoubleX& img2) {
  return(img1.width()==img2.width() && 
      img1.height()==img2.height() && 
      img1.num_channels()==img2.num_channels());
}

bool OpticalFlow::matchDimension(const MCImageDoubleX& img, int width, int height, int nchannels) {
  return(img.width()==width && 
      img.height()==height && 
      img.num_channels()==nchannels);
}

void OpticalFlow::genInImageMask(MCImageDoubleX& mask, const MCImageDoubleX& vx, const MCImageDoubleX& vy, int interval) {
  int width, height;
  width = vx.width();
  height = vx.height();
  if (matchDimension(mask, vx) == false)
  mask.Resize(width, height, 1);
  mask.fill(0);
  double x, y;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      y = vx(j, i, 0) + i;
      x = vy(j, i, 0) + j;
      if (x<interval || x>width - 1 - interval || y<interval || y>height - 1 - interval)
        continue;
      mask(j, i, 0) = 1;
    }
  }
}

void OpticalFlow::genInImageMask(MCImageDoubleX &mask, const MCImageDoubleX &flow, int interval) {
  int imWidth, imHeight;
  imWidth = flow.width();
  imHeight = flow.height();
  if (matchDimension(mask, imWidth, imHeight, 1) == false)
    mask.Resize(imWidth, imHeight, 1);
  else
    mask.fill(0);
  
  //double x, y;

  for (int i = 0; i < imHeight; i++) {
  for (int j = 0; j < imWidth; j++) {
    // not yet
  }
  }
}

void OpticalFlow::Laplacian(MCImageDoubleX& output, const MCImageDoubleX& input, const MCImageDoubleX& weight) {
  if (matchDimension(input, output) == false)
    output.Resize(input.width(), input.height(), 1);
  output.fill(0);
  
    
  if (matchDimension(input, weight) == false) {
  cout<<"Error in image dimension matching Laplacian()!"<<endl;
  return;
  }

  int width = input.width();
  int height = input.height();
  int nChannels = input.num_channels();
  MCImageDoubleX foo(width, height, nChannels);
  foo.fill(0);

  // Horizontal filtering
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width - 1; j++) {
      foo(j, i, 0) = (input(j + 1, i, 0) - input(j, i, 0)) * weight(j, i, 0);
    }
  }

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (j < width - 1) 
      output(j, i, 0) -= foo(j, i, 0);
      if (j > 0)
      output(j, i, 0) += foo(j - 1, i, 0);
    }
  }
  foo.fill(0);

  // Vertical filtering
  for (int i = 0; i < height - 1; i++) {
    for (int j = 0; j < width; j++) {
      foo(j, i, 0) = (input(j, i + 1, 0) - input(j, i, 0)) * weight(j, i, 0);
    }
  }

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (i < height - 1) 
        output(j, i, 0) -= foo(j, i, 0);
      if (i > 0)
        output(j, i, 0) += foo(j, i - 1, 0);
    }
  }
}

void OpticalFlow::estLaplacianNoise(const MCImageDoubleX& Im1, const MCImageDoubleX& Im2, MCImageDoubleX& para) {
  int nChannels = Im1.num_channels();
  int width = Im1.width();
  int height = Im1.height();

  if (para.height() != nChannels) {
    para.Resize(1, 1, nChannels);
    para.fill(0);
	}
  else
    para.fill(0);
  double temp;

  double* total = new double[nChannels];

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      for (int k = 0; k < nChannels; k++) {
        temp = abs(Im1(j, i, k) - Im2(j, i, k));
        if (temp > 0 && temp < 1000000) {
          para(0, 0, k) += temp;
          total[k]++;
        }
      }
    }   
  }
  
  for (int k = 0; k < nChannels; k++) {
    if (total[k] == 0) {
      cout << "All the pixels are invalid in estimation Laplacian noise!!!" << endl;
      cout << "Something severely wrong happened!!!" << endl;
      para(0, 0, k) = 0.001;
    }
    else
      para(0, 0, k) /= total[k];
  }

  delete[] total;
}

void OpticalFlow::BilinearInterpolate(const MCImageDoubleX& pImage,int width,int height,int nChannels,double x,double y,MCImageDoubleX& result, int r, int c) {
	int xx,yy,m,n,u,v,l,offset;
	xx = x;
	yy = y;
	double dx,dy,s;
	dx = max(min(x - xx, 1.), 0.);
	dy = max(min(y - yy, 1.), 0.);

	for(m=0;m<=1;m++) {
		for(n=0;n<=1;n++) {
			u = OpticalFlow::EnforceRange(xx + m, width);
			v = OpticalFlow::EnforceRange(yy + n, height);
			s = fabs(1-m-dx) * fabs(1-n-dy);
			for(l=0 ; l<nChannels ; l++)
				result(c, r, l) += pImage(u, v, l) * s;
		}
	}
}

MCImageDoubleX OpticalFlow::BilinearInterpolate(const MCImageDoubleX& pImage,int width,int height, int nChannels,double x,double y) {
	int xx,yy,m,n,u,v,l,offset;
	xx=x;
	yy=y;
	double dx,dy,s;
	dx=max(min(x-xx,1.),0.);
	dy=max(min(y-yy,1.),0.);

	MCImageDoubleX result(1, 1, nChannels);
	//PixelType<double, nChannels> result;
	
	for(m=0;m<=1;m++)
		for(n=0;n<=1;n++)
		{
			u = EnforceRange(xx+m,width);
			v = EnforceRange(yy+n,height);
			s=fabs(1-m-dx)*fabs(1-n-dy);
			result(0, 0) += pImage(u, v) * s;
		}
	return result;
}

void OpticalFlow::warpFL(MCImageDoubleX& warpIm2, const MCImageDoubleX& Im1, const MCImageDoubleX& Im2, const MCImageDoubleX& vx, const MCImageDoubleX& vy) {
  int width2 = Im2.width();
  int height2 = Im2.height();
  int nChannels2 = Im2.num_channels();

  if (!matchDimension(warpIm2, Im2)) {
    warpIm2.Resize(width2, height2, nChannels2);
  }
  warpIm2.fill(0);
  
  for (int i = 0; i < height2; i++){
		for (int j = 0; j < width2; j++){
			double x, y;
			y = i + vy(j, i, 0);
			x = j + vx(j, i, 0);
			if (x<0 || x>width2-1 || y<0 || y>height2-1) {
				for (int k = 0; k < nChannels2; k++) {
					warpIm2(j, i, k) = Im1(j, i, k);
				}
				continue;
			}
			// Bilinear interpolation
			int xx, yy, m, n, u, v, l;
			xx = x;
			yy = y;
			double dx, dy, s;
			dx = max(min(x - xx, 1.), 0.);
			dy = max(min(y - yy, 1.), 0.);

			for (m = 0; m <= 1; m++) {
				for (n = 0; n <= 1; n++) {
					u = OpticalFlow::EnforceRange(xx + m, width2);
					v = OpticalFlow::EnforceRange(yy + n, height2);
					s = fabs(1 - m - dx) * fabs(1 - n - dy);
					for (l = 0; l < nChannels2; l++) {
						warpIm2(j, i, l) += Im2(u, v, l) * s;
					}
				}
			}
		}
  }
}

void OpticalFlow::ResizeImage(const MCImageDoubleX& pSrcImage,MCImageDoubleX& pDstImage,double Ratio) {
	int DstWidth, DstHeight;
	int srcWidth, srcHeight, Channels;
	
	srcWidth = pSrcImage.width();
	srcHeight = pSrcImage.height();
	Channels = pSrcImage.num_channels();
	DstWidth=(double)srcWidth*Ratio;
	DstHeight=(double)srcHeight*Ratio;
	
	pDstImage.Resize(DstWidth, DstHeight, Channels);
	pDstImage.fill(0);
	
	for(int i=0;i<DstHeight;i++)
		for(int j=0;j<DstWidth;j++)
		{
	    double x,y;

			x=(double)(j+1)/Ratio-1;
			y=(double)(i+1)/Ratio-1;

			// bilinear interpolation
			BilinearInterpolate(pSrcImage,srcWidth,srcHeight,Channels,x,y,pDstImage, i, j);
		}
}

void OpticalFlow::ResizeImage(const MCImageDoubleX& pSrcImage,MCImageDoubleX& pDstImage, int dstWidth, int dstHeight) {
	int srcWidth, srcHeight, nChannels;
  double xRatio,yRatio;
	
	srcWidth = pSrcImage.width();
	srcHeight = pSrcImage.height();
	nChannels = pSrcImage.num_channels();
  xRatio = (double)dstWidth/srcWidth;
  yRatio = (double)dstHeight/srcHeight;
	
	pDstImage.Resize(dstWidth, dstHeight, nChannels);
	pDstImage.fill(0);
	
	for(int i=0;i<dstHeight;i++) {
		for(int j=0;j<dstWidth;j++) {
      double x, y;
			x=(double)(j+1)/xRatio-1;
			y=(double)(i+1)/yRatio-1;

			// bilinear interpolation
			BilinearInterpolate(pSrcImage,srcWidth,srcHeight,nChannels,x,y,pDstImage, i, j);
		}
	}
}

//assume that all images have three channels. 
void OpticalFlow::desaturate(const MCImageDoubleX& im, MCImageDoubleX& out, ColorType color_type) {
  int w = im.width();
  int h = im.height();
  //int ch = im.num_channels();
  out.Resize(w,h,1);
  out.fill(0);

  for (int i=0; i<h; i++) {
    for (int j=0; j<w; j++) {
      if (color_type == RGB)
        out(j, i, 0) = im(j,i,0)*.299+im(j,i,1)*.587+im(j,i,2)*.114;
      else if (color_type == BGR)
        out(j, i, 0) = im(j,i,0)*.114+im(j,i,1)*.587+im(j,i,2)*.299;
    }
  }

}

}
