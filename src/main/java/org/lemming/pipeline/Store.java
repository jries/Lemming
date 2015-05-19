package org.lemming.pipeline;

public interface Store<DataType extends Element> {
	/**
	 * Adds the element el to the store.
	 * 
	 * @param el is the element to be put
	 */
	public void put(DataType el);
	
	/**
	 * Retrieves (and typically removes) an element from the store.
	 * @return element 
	 */
	public DataType get();
	
	/**
	 * Retrieves an element from the store without removing it.
	 * @return element 
	 */
	public DataType peek();
	
	/**
	 * The store is empty.	 * 
	 * @return Checks if the store is empty.
	 */
	boolean isEmpty();
	
	public int getLength();
}
