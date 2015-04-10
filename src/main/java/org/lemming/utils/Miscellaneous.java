package org.lemming.utils;

import org.lemming.outputs.ShowMessage;

/**
 * @author Ronny Sczech
 *
 */
public class Miscellaneous {
	
	/** Generates a list of pixel (x,y) values that surround a fluorophore.
	 *  The size of the window in the x dimension is 2*(5*<code>sigmaX</code>) and 
	 *  in the y dimension is 2*(5*<code>sigmaX</code>*<code>aspectRatio</code>), 
	 *  i.e., a window of 5 sigma. 
	 * 
	 * @param x0 - the x position of the centroid
	 * @param y0 - the y position of the centroid
	 * @param imageWidth - the width of the image
	 * @param imageHeight - the height of the image
	 * @param sigmaX - the sigma value, in the x-direction, for a 2D Gaussian distribution
	 * @param aspectRatio - the aspect ratio for a 2D Gaussian, i.e. sigmaY/sigmaX
	 * @return X - a list of pixels that surrounds (x0,y0) */
	public static int[][] getWindowPixels(int x0, int y0, int imageWidth, int imageHeight, double sigmaX, double aspectRatio){
		
		// Make sure that (x0, y0) is within the image
		if(x0 > imageWidth || y0 > imageHeight) {
			new ShowMessage(String.format("Warning, localization not within image. Got (%d,%d), image size is (%d,%d)", x0, y0, imageWidth, imageHeight));
			return null;
		}
		
		// Automatically select a window around the fluorophore based on the 
		// sigmax and sigmay (ie. the aspect ratio) values.
		// 5*sigma means that 99.99994% of the fluorescene from a simulated 
		// fluorophore (for a Gaussian PSF) is within the specified window.
		// Also, the window has to be at least 1 x 1 pixel
		int halfWindowX = Math.max(1, (int) Math.round(sigmaX*5.0)); 
		int halfWindowY = Math.max(1, (int) Math.round(sigmaX*aspectRatio*5.0));
		
		// make sure that the window remains within the image
		int x1 = Math.max(0, x0 - halfWindowX);
		int y1 = Math.max(0, y0 - halfWindowY);
		int x2 = Math.min(imageWidth - 1, x0 + halfWindowX);
		int y2 = Math.min(imageHeight - 1, y0 + halfWindowY);
		
		// insert the (x,y) window pixel coordinates into the X array
		int size = (x2-x1+1)*(y2-y1+1);
		int[][] X = new int[size][2];
		for(int x=x1, y=y1, i=0; i<size; i++){
			X[i][0] = x;
			X[i][1] = y;
			if (x==x2){
				x = x1;
				y++;
			} else {
				x++;
			}
		}
		return X;
	}
	
}
