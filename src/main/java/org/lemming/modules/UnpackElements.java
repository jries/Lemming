package org.lemming.modules;

import java.util.List;

import org.lemming.interfaces.Element;
import org.lemming.interfaces.Store;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.SingleRunModule;

public class UnpackElements extends SingleRunModule {

	private long start;
	private int counter;
	private Store output;

	public UnpackElements() {
	}
	
	@Override
	protected void beforeRun(){
		start = System.currentTimeMillis();
		output = outputs.values().iterator().next();
	}

	@Override
	public Element process(Element data) {
		if (data instanceof FrameElements){
			FrameElements el = (FrameElements) data;
			counter++;
			List<Element> list = el.getList();
			if(el.isLast()){
				running = false;
				Element last = list.remove(list.size()-1);
				last.setLast(true);
				for (Element l :list)
					output.put(l);
				output.put(last);
			}
			
			for (Element l :list)
				output.put(l);
		} else if (data instanceof Localization){
			counter++;
			if (data.isLast()){
				running = false;
				data.setLast(true);
			}
			output.put(data);
		}
			
		return data;
	}
	
	@Override
	protected void afterRun(){
		System.out.println("Unpacking of " + counter +" elements done in " + (System.currentTimeMillis()-start) + "ms.");
	}

	@Override
	public boolean check() {
		// TODO Auto-generated method stub
		return false;
	}

}
