package org.lemming.interfaces;


/**
 * A Well when run reads and processes from the associated Store.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 * @param <T> - data type
 */
public interface Well<T> extends Runnable {
	
	/**
	 * @param s - Store to put
	 */
	public void setInput(Store<T> s);
}
