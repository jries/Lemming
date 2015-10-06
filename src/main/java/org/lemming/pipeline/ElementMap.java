package org.lemming.pipeline;

import java.util.AbstractMap;
import java.util.Collections;
import java.util.Set;

import org.lemming.interfaces.Element;

public class ElementMap extends AbstractMap<String,Number> implements Element {
	
	private boolean isLast;
	private Set<Entry<String, Number>> entrySet;
	
	public ElementMap(Set<Entry<String, Number>> set){
		this.entrySet = set;
	}


	@Override
	public boolean isLast() {
		return isLast;
	}


	@Override
	public void setLast(boolean b) {
		isLast = b;		
	}

	@Override
	public Set<Entry<String, Number>> entrySet() {
        return Collections.unmodifiableSet(entrySet);
	}

	
}
