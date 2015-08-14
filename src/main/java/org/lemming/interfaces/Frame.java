package org.lemming.interfaces;


import net.imglib2.RandomAccessibleInterval;

public interface Frame<T> extends Element, Comparable<Frame<T>> {
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
