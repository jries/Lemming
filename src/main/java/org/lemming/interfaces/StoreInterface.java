package org.lemming.interfaces;

import java.util.Collection;


public interface StoreInterface<E extends Element> {
	/**
	 * Adds the element el to the store.
	 * @param el is the element to be put
	 */
	public void put(E element);
	
	/**
	 * Retrieves (and typically removes) an element from the store.
	 * @return element 
	 */
	public E get();
	
	/**
	 * Retrieves an element from the store without removing it.
	 * @return element 
	 */
	public E peek();
	
	/**
	 * Checks if the store is empty. 
	 * @return The store is empty.
	 */
	public boolean isEmpty();
	
	/**
	 * Length of the store.
	 * @return length.
	 */
	public int getLength();
	
	/**
	 * Current view of store.
	 * @return view.
	 */
	public Collection<Element> view();
}
