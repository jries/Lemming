package org.lemming.modules;

import java.util.List;

import org.lemming.interfaces.Element;
import org.lemming.pipeline.FrameElements;
import org.lemming.pipeline.Localization;
import org.lemming.pipeline.SingleRunModule;

/**
 * unpack localizations from a frame dependent container
 * 
 * @author Ronny Sczech
 *
 * @param <T> data type
 */
public class UnpackElements<T> extends SingleRunModule {

	private int counter;

	public UnpackElements() {
	}
	

	@SuppressWarnings("unchecked")
	@Override
	public Element processData(Element data) {
		if (data instanceof FrameElements){
			FrameElements<T> el = (FrameElements<T>) data;
			counter++;
			List<Element> list = el.getList();
			if(el.isLast()){
				running = false;
				if (!list.isEmpty()){
					Element last = list.remove(list.size()-1);
					last.setLast(true);
					for(Element l:list) newOutput(l);
					newOutput(last);
				} else {
					newOutput(el);
				}
				return data;
			}

			for(Element l:list) newOutput(l);
		} else if (data instanceof Localization){
			counter++;
			if (data.isLast()){
				running = false;
				data.setLast(true);
			}
			newOutput(data);
		}
		return data;
	}
	
	@Override
	protected void afterRun(){
		System.out.println("Unpacking of " + counter +" elements done in " + (System.currentTimeMillis()-start) + "ms.");
	}

	@Override
	public boolean check() {
		return inputs.size()==1;
	}

}
