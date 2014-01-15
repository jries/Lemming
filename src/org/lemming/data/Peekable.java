package org.lemming.data;

/**
 * The Peekable interface will tag stores that can be peeked. Peeking creates a new Store referencing the same queue.
 * 
 * @author Thomas Pengo, Joe Borbely
 *
 * @param <T>
 */
public interface Peekable<T> {
	
	public Store<T> newPeek();

}
