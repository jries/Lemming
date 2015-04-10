package org.lemming.interfaces;


/**
 * The Peekable interface will tag stores that can be peeked. Peeking creates a new Store referencing the same queue.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 * @param <T> data type
 */
public interface Peekable<T> {
	
	/**
	 * @return Store
	 */
	public Store<T> newPeek();

}
