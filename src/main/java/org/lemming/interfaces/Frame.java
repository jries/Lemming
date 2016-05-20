package org.lemming.interfaces;

import net.imglib2.RandomAccessibleInterval;

/**
 * Frame holds an 2D image plus some useful metadata.
* @author Ronny Sczech
*
* @param <T> - data type
*/
public interface Frame<T> extends Element, Comparable<Frame<T>> {
	/** 
	 * @return Return the frame number in the movie */
	long getFrameNumber();
	
	/**  
	 * @return Return the pixel values of this frame */
	RandomAccessibleInterval<T> getPixels();
	
	/**  
	 * @return Return the width of this frame */
	int getWidth();
	
	/**  
	 * @return Return the height of this frame */
	int getHeight();
	
	/**  
	 * @return Return the pixel size */
	double getPixelDepth();

}

