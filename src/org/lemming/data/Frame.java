package org.lemming.data;

public interface Frame<T> {
	
	/** Return the frame number in the movie */
	public long getFrameNumber();
	
	/** Return the pixel values of this frame */
	public T getPixels();
	
	/** Return the width of this frame */
	public int getWidth();
	
	/** Return the height of this frame */
	public int getHeight();
}
