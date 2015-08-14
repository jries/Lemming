package org.lemming.modules;

import java.beans.BeanInfo;
import java.beans.IntrospectionException;
import java.beans.Introspector;
import java.beans.PropertyDescriptor;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.lemming.interfaces.Element;
import org.lemming.interfaces.Store;
import org.lemming.pipeline.ExtendableTable;
import org.lemming.pipeline.SingleRunModule;

public class DataTable extends SingleRunModule {

	private Store store;
	private ExtendableTable table;
	private List<PropertyDescriptor> descriptors;
	private long start;

	public DataTable() {
		table = new ExtendableTable();
		descriptors = new ArrayList<>();
	}
	
	@Override
	public void beforeRun(){
		store = inputs.get(iterator);
		start = System.currentTimeMillis();
		while (store.isEmpty())
			pause(10);
		Element el = store.peek();
		try {
			BeanInfo b = Introspector.getBeanInfo(el.getClass());
			for (PropertyDescriptor p : b.getPropertyDescriptors()) {
				String prop = p.getName();
				boolean test = prop.contains("class") | prop.contains("last");
				if (!test){
					if (!table.columnNames().contains(prop)){
						table.addNewMember(prop);
						descriptors.add(p);
					}
						
				}
			}
		} catch (IntrospectionException e) {
			e.printStackTrace();
		}
	}

	
	public Element process(Element data) {
		if (data==null) return null;
		if (data.isLast()) cancel();
		Map<String,Object> row = new HashMap<>();
		for (PropertyDescriptor p:descriptors){
			try {
				row.put(p.getName(), p.getReadMethod().invoke(data));
			} catch ( IllegalAccessException | IllegalArgumentException | InvocationTargetException e) {
				e.printStackTrace();
			}
		}
		table.addRow(row);
		return null;
	}
	
	@Override
	protected void afterRun() {
		System.out.println("Import in DataTable done in " + (System.currentTimeMillis() - start) + "ms.");
	}
	
	public ExtendableTable getTable(){
	 return table;	
	}

	@Override
	public boolean check() {
		// TODO Auto-generated method stub
		return false;
	}

}
