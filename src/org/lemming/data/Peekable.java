package org.lemming.data;

public interface Peekable<T> {
	
	public Store<T> newPeek();

}
