package org.lemming.interfaces;

/**
 * This interface sets a boolean that is checkable by all inherited objects
 * 
 * @author Ronny Sczech
 */

public interface Element{
	/**
	 * @return checks if Element is the last in the queue.
	 */
	boolean isLast();
	
	/**
	 * @param last - set this Element object as last
	 */
	void setLast(boolean last);
}
