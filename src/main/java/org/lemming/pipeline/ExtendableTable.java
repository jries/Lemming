package org.lemming.pipeline;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import javolution.util.FastTable;

/**
 * @author Ronny Sczech
 *
 */
public class ExtendableTable {
	
	private int nRows = 0;
	private Map<String, List<Object>> table = new HashMap<>();
	
	/**
	 * 
	 */
	public ExtendableTable(){
		addLocalizationMembers();
	}
	
	/**
	 * @param member - member
	 */
	public void addNewMember(String member) {
		table.put(member,new FastTable<>());
	}
	
	/**
	 * 
	 */
	public void addLocalizationMembers(){
		//table.put("id",new FastTable<Object>());
		addNewMember("xpix");
		addNewMember("ypix");
	}
	
	
	public Set<String> columnNames(){
		return table.keySet();
	}
	
	/**
	 * @param row - row
	 */
	public void addRow(Map<String,Object> row){
		for (Entry<String,Object> e : row.entrySet()){
			add(e.getKey(),e.getValue());
		}
		nRows++;
	}
	
	/**
	 * @param row - row
	 * @return data
	 */
	public Map<String,Object> getRow(int row){
		Map<String,Object> map = new HashMap<>(); // row map
		for (String key : table.keySet())
			map.put(key, table.get(key).get(row));
		return map;
	}
	
	/**
	 * @param col - colummn
	 * @return column
	 */
	public List<Object> getColumn(String col){
		List<Object> c = table.get(col);
		if(c==null)
			System.err.println("unknown column");
		return c;
	}
	
	/**
	 * @param col - colummn
	 * @return column
	 */
	public Object getData(String col, int row){
		List<Object> c = table.get(col);
		if(c != null && row < nRows)
			return c.get(row);
		
		System.err.println("unknown column or row");
		return null;
	}
	
	/**
	 * @param member - member 
	 * @param o - object
	 */
	public void add(String member, Object o){
		List<Object> t = table.get(member);
		if (t!=null){
			t.add(o);
			return;
		}
		System.err.println("unknown column");
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
	public Store getFIFO() {
		
		return new Store () {
			int lastRow = 0;
						
			@Override
			public boolean isEmpty() {
				return lastRow >= nRows;
			}
			
			@SuppressWarnings("unchecked")
			@Override
			public void put(Element el) {
				if (el instanceof Map){
					Map<String,Object> em = (Map<String,Object>) el;
					addRow(em);
					nRows++;
				}
			}

			@Override
			public Element get() {
				ElementMap em = new ElementMap(getRow(lastRow++).entrySet());
				if (isEmpty())	
					em.setLast(true);
				return em;
			}

			@Override
			public Element peek() {
				return new ElementMap(getRow(lastRow).entrySet());
			}

			@Override
			public int getLength() {
				return lastRow;
			}

			@Override
			public Collection<Element> view() {
				return this.view();
			}			

		};
	}
	
}
