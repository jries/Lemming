package org.lemming.pipeline;

import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.lemming.interfaces.Element;
import org.lemming.interfaces.Store;

import javolution.util.FastTable;
import javolution.util.function.Predicate;

/**
 * @author Ronny Sczech
 *
 */
public class ExtendableTable {
	
	private int nRows = 0;
	private Map<String, List<Number>> table = new LinkedHashMap<>();
	public Map<String, Predicate<Number>> filtersCollection = new HashMap<>();
	private Map<String, String> names = new LinkedHashMap<>();
	
	/**
	 * 
	 */
	public ExtendableTable(){
	}
	
	public ExtendableTable(Map<String, List<Number>> table){
		this.table = table;
		for(String key : table.keySet())
			names.put(key,key);
	}
	
		
	/**
	 * @param member - member
	 */
	public void addNewMember(String member) {
		table.put(member,new FastTable<Number>());
		names.put(member,member);
	}
	
	/**
	 * 
	 */
	public void addXYMembers(){
		//table.put("id",new FastTable<Object>());
		addNewMember("x");
		addNewMember("y");
	}
	
	
	public Set<String> columnNames(){
		return table.keySet();
	}
	
	public ExtendableTable filter(){
		
		if (filtersCollection.isEmpty()) return this;
		
		final ExtendableTable filteredTable = new ExtendableTable(); //new instance
		for (String col: this.columnNames())
			filteredTable.addNewMember(col);
		
		Map<String, Number> row;
		for (int index = 0 ; index < getNumberOfRows(); index++){
			row = getRow(index);
			boolean filtered = true;
			for (String key : filtersCollection.keySet()){
				Number value = row.get(key);
				if (value!=null)
					filtered = filtered && (filtered == filtersCollection.get(key).test(value));
				else
					filtered = false;
			}
			if(filtered)
				filteredTable.addRow(row);
		}
		return filteredTable;
	}
	
	public void addFilterMinMax(final String col, final double min, final double max){
		Predicate<Number> p = new Predicate<Number>(){

			@Override
			public boolean test(Number t) {				
				return (t.doubleValue()>=min) && (t.doubleValue()<=max);
			}
			
		};
		filtersCollection.put(col, p);
	}
	
	public void addFilterExact(final String col, final Number o){
		Predicate<Number> p = new Predicate<Number>(){

			@Override
			public boolean test(Number t) {	
				return t.equals(o);
			}
			
		};
		filtersCollection.put(col, p);
	}
	
	/**
	 * @param row - row
	 */
	public void addRow(Map<String,Number> row){
		for (Entry<String,Number> e : row.entrySet()){
			add(e.getKey(),e.getValue());
		}
		nRows++;
	}
	
	/**
	 * @param row - row
	 * @return data
	 */
	public Map<String, Number> getRow(int row){
		Map<String,Number> map = new HashMap<>(); // row map
		for (String key : table.keySet())
			map.put(key, table.get(key).get(row));
		return map;
	}
	
	/**
	 * @param col - colummn
	 * @return column
	 */
	public List<Number> getColumn(String col){
		return table.get(col);
	}
	
	/**
	 * @param col - colummn
	 * @return column
	 */
	public Object getData(String col, int row){
		List<Number> c = table.get(col);
		if(c != null && row < nRows)
			return c.get(row);
		
		System.err.println("unknown column or row");
		return null;
	}
	
	/**
	 * @param member - member 
	 * @param o - object
	 */
	public void add(String member, Number o){
		List<Number> t = table.get(member);
		if (t!=null){
			if (t.size() == nRows) 
				nRows++;
			t.add(o);
			return;
		}
		System.err.println("unknown column");
	}
	
	
	/**
	 * @return number of rows
	 */
	public int getNumberOfRows() {
		if (nRows < table.values().iterator().next().size())
			nRows = table.values().iterator().next().size();
		return nRows;
	}
	
	/**
	 * @return names
	 */
	public Map<String, String> getNames() {
		return names;
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
				return lastRow >= getNumberOfRows();
			}
			
			@SuppressWarnings("unchecked")
			@Override
			public void put(Element el) {
				if (el instanceof Map){
					Map<String,Number> em = (Map<String,Number>) el;
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
