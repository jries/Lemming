package org.lemming.interfaces;

/**
 * @author Ronny Sczech
 *
 * @param <T> data type
 */
public interface Splitter<T> extends Well<T> {
	
	/**
	 * @param s - the specified Store
	 */
	public void addOutput(Store<T> s);
	
}
