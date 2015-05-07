package org.lemming.pipeline;

public interface Element {
	/**
	 * @return checks if Element is the last in the queue.
	 */
	public boolean isLast();
	
	public void setLast(boolean b);
}
