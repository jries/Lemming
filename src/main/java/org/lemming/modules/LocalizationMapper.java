package org.lemming.modules;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.MapElement;
import org.lemming.pipeline.SingleRunModule;

public class LocalizationMapper extends SingleRunModule {

	private long start;

	public LocalizationMapper() {
		
	}
	@Override
	public void beforeRun() {
		start = System.currentTimeMillis();
	}
	@Override
	public boolean check() {
		return inputs.size()==1;
	}

	@Override
	public Element processData(Element data) {
		try{
			MapElement me = (MapElement) data;
			long col1 = (long) me.get().get("col1");
			double col2 = (double) me.get().get("col2");
			double col3 = (double) me.get().get("col3");
			Localization loc = new Localization(col1, col2, col3);
			return loc;
		} catch (ClassCastException | NullPointerException e){
			return null;
		}
	}
	
	@Override
	public void afterRun() {
		System.out.println("Mapping data done in " + (System.currentTimeMillis()-start) + "ms.");
	}
}
