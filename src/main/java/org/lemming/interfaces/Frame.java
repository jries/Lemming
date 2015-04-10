package org.lemming.interfaces;

import net.imglib2.RandomAccessibleInterval;

/**
 * Interface representing a Frame. A Frame is the input to a localization module and typically contains a 2D image, but being generic for T, it can 
 * in principle be anything.
 * 
 * @author Thomas Pengo, Joe Borbely
 * @param <T> data type
 */
public interface Frame<T> extends Element{
	
	/** 
	 * @return Return the frame number in the movie */
	public long getFrameNumber();
	
	/**  
	 * @return Return the pixel values of this frame */
	public RandomAccessibleInterval<T> getPixels();
	
	/**  
	 * @return Return the width of this frame */
	public int getWidth();
	
	/**  
	 * @return Return the height of this frame */
	public int getHeight();
}
