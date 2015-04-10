package org.lemming.interfaces;

/**
 * A basic X,Y localization with an ID.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 */
public interface Localization extends Element{
	/**
	 * @return ID
	 */
	public long getID();
	
	/**
	 * @return x
	 */
	public double getX();
	/**
	 * @return y
	 */
	public double getY();	
}
