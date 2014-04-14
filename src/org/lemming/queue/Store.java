package org.lemming.queue;

/**
 * Store represents a repository of generic objects. Only a get, a put and an isempty class are provided.
 * 
 * @author Thomas Pengo
 *
 * @param <DataType>
 */
public interface Store<DataType> {

	/**
	 * Adds the element el to the store.
	 * 
	 * @param el is the element to be put
	 */
	public void put(DataType el);
	
	/**
	 * Retrieves (and typically removes) an element from the store.
	 * 
	 * @return
	 */
	public DataType get();
	
	/**
	 * The store is empty.
	 * 
	 * @return
	 */
	boolean isEmpty();
}
