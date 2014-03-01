package org.lemming.data;

import net.imglib2.RandomAccessibleInterval;

/**
 * Interface representing a Frame. A Frame is the input to a localization module and typically contains a 2D image, but being generic for T, it can 
 * in principle be anything.
 * 
 * @author Thomas Pengo, Joe Borbely
 * 
 *  
 * @param <T>
 */
public interface Frame<T> {
	
	/** Return the frame number in the movie */
	public long getFrameNumber();
	
	/** Return the pixel values of this frame */
	public RandomAccessibleInterval<T> getPixels();
	
	/** Return the width of this frame */
	public int getWidth();
	
	/** Return the height of this frame */
	public int getHeight();
}
