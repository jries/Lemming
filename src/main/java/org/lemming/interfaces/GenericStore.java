package org.lemming.interfaces;

import java.util.Collection;


public interface GenericStore<E extends Element> {
	/**
	 * Adds the element el to the store.
	 * 
	 * @param el is the element to be put
	 */
	public void put(E el);
	
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
	 * The store is empty.	 * 
	 * @return Checks if the store is empty.
	 */
	public boolean isEmpty();
	
	public int getLength();

	public Collection<Element> view();
}
