package org.lemming.modules;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.ElementMap;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.SingleRunModule;

/**
 * 
 * 
 * @author Ronny Sczech
 *
 */
public class LocalizationMapper extends SingleRunModule {

	private final String[] nameArray;

	public LocalizationMapper(String[] nameArray) {
		this.nameArray = nameArray;
	}

	@Override
	public boolean check() {
		return inputs.size()==1 && nameArray.length>3;
	}

	@Override
	public Element processData(Element data) {
		try{
			ElementMap me = (ElementMap) data; 
			double col1 = (Double) me.get(nameArray[0]);
			double col2 = (Double) me.get(nameArray[1]);
			double col3 = (Double) me.get(nameArray[2]);
			long col4 = (Long) me.get(nameArray[3]);
			return new Localization(col1, col2, col3,col4);
		} catch (ClassCastException | NullPointerException e){
			return null;
		}
	}
	
	@Override
	public void afterRun() {
		System.out.println("Mapping data done in " + (System.currentTimeMillis()-start) + "ms.");
	}
}
