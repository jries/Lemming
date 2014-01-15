package org.lemming.outputs;

import java.util.Timer;
import java.util.TimerTask;

import ij.ImagePlus;
import ij.process.FloatProcessor;

import org.lemming.data.Localization;
import org.lemming.data.Store;
import org.lemming.interfaces.Rendering;
import org.lemming.utils.Functions;
import org.lemming.utils.Miscellaneous;

/**
 * Renders the localizations received in the input using Gaussians. 
 * 
 * The display is updated asynchronously at regular intervals (100 ms).
 * 
 * @author Joe Borbely
 *
 */
public class GaussRenderOutput extends SI<Localization> implements Rendering {

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

	ImagePlus ip; FloatProcessor fp;
	Timer t = new Timer();

	@Override
	public void run() {
		pixels = new double[width*height];
		fp = new FloatProcessor(width, height, pixels);
		ip = new ImagePlus(title, fp);
		ip.show();
		
		t.schedule(new TimerTask() {
			@Override
			public void run() {
				update();
			}
		}, 100, 100);		
				
		super.run();
	}

	double maxVal=-Float.MAX_VALUE; // keeps track of the maximum value in the histogram

	@Override
	public void process(Localization loc) {
		double x = loc.getX();
		double y = loc.getY();
		int[][] X = Miscellaneous.getWindowPixels((int)x, (int)y, width, height, sigmaX, aspectRatio);
		if(X==null) return;
		double[] Params = {background, x, y, area, theta, sigmaX, aspectRatio};
		double[] fcn = Functions.gaussian2D(X, Params);
		for (int i=0, j=X.length, idx; i<j; i++){
			idx = X[i][0] + X[i][1]*width;
			pixels[idx] += fcn[i];
			if (pixels[idx] > maxVal)
				maxVal = pixels[idx]; 
			fp.setf(idx, (float)pixels[idx]);				
		}
		
	}
	
	void update() {
        if (ip==null)
        	return;
        
        ip.updateAndDraw();
		ip.setDisplayRange(0, maxVal);	
	}


}
