package org.lemming.outputs;

import ij.ImagePlus;
import ij.process.FloatProcessor;

import org.lemming.data.Localization;
import org.lemming.data.Store;
import org.lemming.interfaces.Output;
import org.lemming.utils.Functions;
import org.lemming.utils.Miscellaneous;

public class GaussRenderOutput implements Output {

	Store<Localization> localizations;
	double[] pixels;

	// set default values for rendering
	int width = 256; // image width
	int height = 256; // image height	
	double area = 1000.0; // total area under the Gaussian curve
	double background = 0.0; // background signal
	double aspectRatio = 1.0; // aspect ratio of the sigma's, sigmay/sigmax
	double sigmaX = 2.0; // sigma of the 2D Gaussian in the x-direction
	double theta = 0.0; // rotation angle, in radians, between a fluorescing molecule and the image canvas	
	String title = "LemMING!"; // title of the image

	/** Draw an image from a Store of molecule localizations and render each
	 * molecule as a 2-dimensional Gaussian. This construct uses default variables
	 * for the image width and height, for the Gaussian parameters and for the
	 * image title.
	 * @see #GaussRenderOutput(int, int) GaussRenderOutput(width, height)
	 * @see #GaussRenderOutput(int, int, double, double, double, double, double, String) GaussRenderOutput(width, height, area, background, aspectRatio, sigmaX, theta, title) 
	 * @see {@link org.lemming.utils.Functions.gaussian2D} */
	public GaussRenderOutput() {}
	
	/** Draw an image from a Store of molecule localizations and render each
	 * molecule as a 2-dimensional Gaussian. This construct uses default values
	 * for the Gaussian parameters and for the image title.
	 * @param width - the image width, in pixels (e.g. 256)
	 * @param height - the image height, in pixels (e.g. 256) 
	 * @see #GaussRenderOutput() 
	 * @see #GaussRenderOutput(int, int, double, double, double, double, double, String) GaussRenderOutput(width, height, area, background, aspectRatio, sigmaX, theta, title) 
	 * @see {@link org.lemming.utils.Functions.gaussian2D} */
	public GaussRenderOutput(int width, int height) {
		this.width = width;
		this.height = height;
	}

	/** Draw an image from a Store of molecule localizations and render each
	 * molecule as a 2-dimensional Gaussian.
	 * @param width - the image width, in pixels (e.g. 256)
	 * @param height - the image height, in pixels (e.g. 256)
	 * @param area - the area (total intensity) of a normalized, Gaussian function
	 * @param background - the background signal
	 * @param aspectRatio - the aspect ratio, i.e., sigmaY/sigmaX
	 * @param sigmaX - the Gaussian sigma value in the x-direction
	 * @param theta - the rotation angle, in radians, between the fluorescing molecule and the image canvas
	 * @param title - the title to display in the image Window 
	 * @see #GaussRenderOutput()
	 * @see #GaussRenderOutput(int, int) */
	public GaussRenderOutput(int width, int height, double area, double background, double aspectRatio, double sigmaX, double theta, String title) {
		this.width = width;
		this.height = height;
		this.area = area;
		this.background = background;
		this.aspectRatio = aspectRatio;
		this.sigmaX = sigmaX;
		this.theta = theta;
		this.title = title;
	}	

	/** Set the parameters for the normalized, 2-dimensional, Gaussian function 
	 * that is used to render a molecule.
	 * @param area - the area (total intensity) of a normalized, Gaussian function
	 * @param background - the background signal
	 * @param aspectRatio - the aspect ratio, i.e., sigmaY/sigmaX
	 * @param sigmaX - the Gaussian sigma value in the x-direction
	 * @param theta - the rotation angle, in radians, between the fluorescing molecule and the image canvas 
	 * @see {@link org.lemming.utils.Functions.gaussian2D} */
	void setGaussianParameters(double area, double background, double aspectRatio, double sigmaX, double theta){
		this.area = area;
		this.background = background;
		this.aspectRatio = aspectRatio;
		this.sigmaX = sigmaX;
		this.theta = theta;
	}
	
	/** Set the image title to display on Titlebar of the image Window. 
	 * @param title - the image title */
	void setTitle(String title){
		this.title = title;
	}

	@Override
	public void run() {
		
		if (localizations==null) {new NullStoreWarning(this.getClass().getName()); return;}
		
		pixels = new double[width*height];
		FloatProcessor fp = new FloatProcessor(width, height, pixels);
		ImagePlus ip = new ImagePlus(title, fp);
		ip.show();
		
		Localization loc;
		while ((loc=localizations.get())!=null){
			double x = loc.getX();
			double y = loc.getY();
			int[][] X = Miscellaneous.getWindowPixels((int)x, (int)y, width, height, sigmaX, aspectRatio);
			if(X==null) return;
			double[] Params = {background, x, y, area, theta, sigmaX, aspectRatio};
			double[] fcn = Functions.gaussian2D(X, Params);
			double val, maxval=-Double.MAX_VALUE; // maxval is used for setting the ImagePlus display range
			for (int i=0, j=X.length, idx; i<j; i++){
				idx = X[i][0] + X[i][1]*width;
				pixels[idx] += fcn[i];
				val = pixels[idx];
				if (val > maxval)
					maxval = val; 
				fp.setf(idx, (float)val);				
			}
			ip.updateAndDraw();
			ip.setDisplayRange(0, maxval);
		}
	}

	@Override
	public void setInput(Store<Localization> s) {
		localizations = s;
	}

}
