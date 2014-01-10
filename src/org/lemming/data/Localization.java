package org.lemming.data;

/**
 * A basic X,Y localization with an ID.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 */
public interface Localization {
	public long getID();
	
	public double getX();
	public double getY();
}
