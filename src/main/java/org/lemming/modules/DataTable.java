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
import org.lemming.pipeline.ExtendableTable;
import org.lemming.pipeline.SingleRunModule;

/**
 * converts a queue into a table
 * 
 * @author Ronny Sczech
 *
 */
public class DataTable extends SingleRunModule {

	private final ExtendableTable table;
	private final List<PropertyDescriptor> descriptors;

	public DataTable() {
		table = new ExtendableTable();
		descriptors = new ArrayList<>();
	}
	
	@Override
	public void beforeRun(){
		start = System.currentTimeMillis();
		Element el = inputs.get(iterator).peek();
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

	
	public Element processData(Element data) {
		if (data==null) return null;
		if (data.isLast()) cancel();
		Map<String,Number> row = new HashMap<>();
		for (PropertyDescriptor p:descriptors){
			try {
				row.put(p.getName(), (Number) p.getReadMethod().invoke(data));
			} catch ( IllegalAccessException | IllegalArgumentException | InvocationTargetException e) {
				e.printStackTrace();
			}
		}
		table.addRow(row);
		return null;
	}
	
	@Override
	protected void afterRun() {
		System.out.println("DataTable imported " + table.getNumberOfRows() +" elements in " + (System.currentTimeMillis() - start) + "ms.");
	}
	
	public ExtendableTable getTable(){
	 return table;	
	}

	@Override
	public boolean check() {
		return inputs.size()==1;
	}

}
