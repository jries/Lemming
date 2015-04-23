package org.lemming.data;

import javolution.util.FastMap;
import javolution.util.FastTable;

import org.lemming.interfaces.Store;

/**
 * @author Ronny Sczech
 *
 */
public class ExtendableTable {
	
	private int nRows = 0;
	private FastMap<String, FastTable<Object>> table = new FastMap<String, FastTable<Object>>();
	
	/**
	 * 
	 */
	public ExtendableTable(){
	}
	
	/**
	 * @param member - member
	 */
	public void addNewMember(String member) {
		table.put(member,new FastTable<Object>());
	}
	
	/**
	 * 
	 */
	public void addLocalizationMembers(){
		table.put("id",new FastTable<Object>());
		table.put("x",new FastTable<Object>());
		table.put("y",new FastTable<Object>());
	}
	
	
	/**
	 * @param row - row
	 * @return row elements as Map
	 */
	public FastMap<String,Object> newRow(){
		FastMap<String,Object> map = new FastMap<String,Object>();
		for (String key : table.keySet()){
			map.put(key,new Object());
		}
		return map;
	}
	
	/**
	 * @param row - row
	 */
	public void addRow(FastMap<String,Object> row){
		for (String key : row.keySet()){
			set(key,row.get(key));
		}
		nRows++;
	}
	
	/**
	 * @param row - row
	 * @return data
	 */
	public FastMap<String,Object> getRow(int row){
		FastMap<String,Object> map = new FastMap<String,Object>(); // row map
		for (String key : table.keySet())
			map.put(key, table.get(key).get(row));
		return map;
	}
	
	/**
	 * @param member - member 
	 * @param o - object
	 */
	public void set(String member, Object o){
		FastTable<Object> t = table.get(member);
		if (t==null) { System.err.println("unknown column"); return;}
		t.add(o);
	}
	
	
	/**
	 * @return number of rows
	 */
	public int getNumberOfRows() {
		return nRows;
	}
	
	
	/**
	 * This method provides a bridge between the Workspace abstraction and the Store abstraction. 
	 * 
	 * It creates a mutable view on the workspace which allows a module working with Stores to have read/write access to the Workspace in a 
	 * first-in-first-out order using the methods provided by the Store interface. 
	 * 
	 * The put method adds the data to the end of the table, the get keeps track of the last row read. 
	 * The get method is NON-BLOCKING: if the table is empty, or you read all rows, it returns 'null'.
	 *  
	 * @return a class implementing the Store interface.
	 */
	public Store<FastMap<String,Object>> getFIFO() {
		return new Store<FastMap<String,Object>> () {
			int lastRow = 0;
						
			@Override
			public boolean isEmpty() {
				return lastRow >= nRows;
			}
			
			@Override
			public FastMap<String,Object> get() {
				if (isEmpty()) {
					FastMap<String, Object> row = getRow(lastRow-1);
					for (String key : row.keySet())
						row.put(key, new LastElement(true));
					return row;
				}
				
				return getRow(lastRow++);
			}
			
			@Override
			public void put(FastMap<String, Object> el) {
				addRow(el);
				nRows++;			
			}			

		};
	}
}
